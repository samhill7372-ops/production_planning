"""Multi-Output Yield Prediction Service for API deployment.

Two-step XGBoost model:
1. yield_model: Predicts overall yield factor (fraction of input → output)
2. ratio_model: Predicts BIN distribution (how output splits across Grade/Length/Width)

Self-contained — prediction functions copied from src/multi_output_prediction.py
so the API deployment has no dependency on the src/ package.
"""

from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Prediction helper functions ───────────────────────────────────────

def _round_bucket(series: pd.Series, bucket: float) -> pd.Series:
    if bucket <= 0:
        return series
    return (series / bucket).round() * bucket


def build_features(
    df_261: pd.DataFrame, round_length: float = 12.0, round_width: float = 1.0
) -> pd.DataFrame:
    order_col = "MANUFACTURINGORDER"
    df_261 = df_261.copy()

    if round_length > 0:
        df_261["TALLYLENGTH"] = _round_bucket(df_261["TALLYLENGTH"], round_length)
    if round_width > 0:
        df_261["TALLYWIDTH"] = _round_bucket(df_261["TALLYWIDTH"], round_width)

    bfin_total = df_261.groupby(order_col)["BFIN"].sum(min_count=1).rename("TOTAL_BFIN")
    bfin_log = np.log1p(bfin_total).rename("LOG1P_TOTAL_BFIN")

    species_ohe = pd.get_dummies(df_261["MATERIALSPECIE"], prefix="SPECIE")
    species_ohe[order_col] = df_261[order_col]
    species_ohe = species_ohe.groupby(order_col).max()

    thickness_ohe = pd.get_dummies(df_261["MATERIALTHICKNESS"], prefix="THICKNESS")
    thickness_ohe[order_col] = df_261[order_col]
    thickness_ohe = thickness_ohe.groupby(order_col).max()

    plant_ohe = pd.get_dummies(df_261["PLANT"], prefix="PLANT")
    plant_ohe[order_col] = df_261[order_col]
    plant_ohe = plant_ohe.groupby(order_col).max()

    grade_counts = (
        df_261.pivot_table(
            index=order_col,
            columns="TALLYGRADE",
            values="BFIN",
            aggfunc="size",
            fill_value=0,
        )
        .add_prefix("INPUT_GRADE_COUNT_")
        .sort_index(axis=1)
    )

    length_counts = (
        df_261.pivot_table(
            index=order_col,
            columns="TALLYLENGTH",
            values="BFIN",
            aggfunc="size",
            fill_value=0,
        )
        .add_prefix("IN_LEN_")
        .sort_index(axis=1)
    )

    width_counts = (
        df_261.pivot_table(
            index=order_col,
            columns="TALLYWIDTH",
            values="BFIN",
            aggfunc="size",
            fill_value=0,
        )
        .add_prefix("IN_WID_")
        .sort_index(axis=1)
    )

    input_size_volume = (
        df_261.pivot_table(
            index=order_col,
            columns=["TALLYLENGTH", "TALLYWIDTH"],
            values="BFIN",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(
            columns=lambda col: f"IN_{col[0]}x{col[1]}_VOLUME"
            if isinstance(col, tuple)
            else f"IN_{col}_VOLUME"
        )
        .sort_index(axis=1)
    )

    features = pd.concat(
        [
            bfin_total,
            bfin_log,
            species_ohe,
            thickness_ohe,
            plant_ohe,
            grade_counts,
            length_counts,
            width_counts,
            input_size_volume,
        ],
        axis=1,
    ).reset_index()
    return features


def align_features(features: pd.DataFrame, template_df: pd.DataFrame) -> tuple:
    bin_cols = [col for col in template_df.columns if col.startswith("BIN_")]
    drop_cols = {"MANUFACTURINGORDER", "TOTAL_BFOUT", "TOTAL_YIELD_FACTOR"}
    feature_cols = [col for col in template_df.columns if col not in bin_cols]
    feature_cols = [col for col in feature_cols if col not in drop_cols]

    total_bfin = float(features["TOTAL_BFIN"].iloc[0])
    aligned = features.reindex(columns=feature_cols, fill_value=0.0)
    return aligned, total_bfin


def align_features_batch(
    features: pd.DataFrame, template_df: pd.DataFrame
) -> tuple:
    """Align features for multiple orders simultaneously.

    Returns:
        aligned: DataFrame (n_orders x n_features)
        total_bfin_values: numpy array of TOTAL_BFIN per order
        order_ids: list of MANUFACTURINGORDER values in row order
    """
    bin_cols = [col for col in template_df.columns if col.startswith("BIN_")]
    drop_cols = {"MANUFACTURINGORDER", "TOTAL_BFOUT", "TOTAL_YIELD_FACTOR"}
    feature_cols = [col for col in template_df.columns if col not in bin_cols]
    feature_cols = [col for col in feature_cols if col not in drop_cols]

    order_ids = features["MANUFACTURINGORDER"].tolist()
    total_bfin_values = features["TOTAL_BFIN"].values.astype(float)
    aligned = features.reindex(columns=feature_cols, fill_value=0.0)
    return aligned, total_bfin_values, order_ids


def normalize_ratios(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.maximum(y_pred, 0.0)
    denom = np.where(
        y_pred.sum(axis=1, keepdims=True) == 0.0,
        1.0,
        y_pred.sum(axis=1, keepdims=True),
    )
    return y_pred / denom


def apply_zero_history_guardrail(
    ratios: np.ndarray,
    template_df: pd.DataFrame,
    input_species: list,
    bin_cols: list,
) -> np.ndarray:
    if not input_species:
        return ratios
    species_cols = [
        f"SPECIE_{val}" for val in input_species if f"SPECIE_{val}" in template_df
    ]
    if not species_cols:
        return ratios
    mask_species = template_df[species_cols].sum(axis=1) > 0
    if not mask_species.any():
        return ratios

    history = (template_df.loc[mask_species, bin_cols] > 0).any(axis=0)
    allowed = history.to_numpy()
    adjusted = ratios.copy()
    adjusted[:, ~allowed] = 0.0
    return normalize_ratios(adjusted)


def max_length_check(
    ratios: np.ndarray, bin_cols: list, max_length: float
) -> np.ndarray:
    adjusted = ratios.copy()
    for idx, bin_name in enumerate(bin_cols):
        if bin_name == "BIN_OTHER":
            continue
        parts = bin_name.split("_", 3)
        if len(parts) != 4:
            continue
        try:
            length_val = float(parts[2])
        except ValueError:
            continue
        if length_val > max_length:
            adjusted[:, idx] = 0.0
    return normalize_ratios(adjusted)


def round_counts_preserve_total(
    counts: np.ndarray, total: float, total_input: float
) -> np.ndarray:
    target = int(round(total))
    target = min(target, int(np.floor(total_input)))
    if target <= 0:
        return np.zeros_like(counts, dtype=int)
    base = np.floor(counts).astype(int)
    remainder = target - int(base.sum())
    if remainder > 0:
        frac = counts - base
        order = np.argsort(-frac)
        base[order[:remainder]] += 1
    elif remainder < 0:
        frac = counts - base
        order = np.argsort(frac)
        for idx in order:
            if remainder == 0:
                break
            if base[idx] > 0:
                base[idx] -= 1
                remainder += 1
    return base


# ─── Service class ─────────────────────────────────────────────────────

class MultiOutputService:
    """Loads the two-step XGBoost model and exposes prediction methods."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        warnings.filterwarnings("ignore", category=UserWarning)

        self.model_bundle = joblib.load(model_dir / "baseline_yield_model.joblib")
        self.template_df = joblib.load(model_dir / "training_template.joblib")
        self.bin_cols = [
            c for c in self.template_df.columns if c.startswith("BIN_")
        ]

    def predict_single_order(self, df_order: pd.DataFrame) -> dict | None:
        """Run prediction for a single manufacturing order.

        Returns dict with keys:
            manufacturing_order, total_bfin, yield_factor,
            predicted_output, total_boards, distribution
        """
        ratio_model = self.model_bundle["ratio_model"]
        yield_model = self.model_bundle["yield_model"]

        if df_order.empty:
            return None

        order_id = str(df_order["MANUFACTURINGORDER"].iloc[0])

        # Build features
        features = build_features(df_order)
        x_aligned, total_bfin = align_features(features, self.template_df)

        # Species info for guardrails
        input_species = (
            df_order["MATERIALSPECIE"].dropna().astype(str).unique().tolist()
            if "MATERIALSPECIE" in df_order.columns
            else []
        )
        max_input_length = (
            float(df_order["TALLYLENGTH"].max())
            if "TALLYLENGTH" in df_order.columns
            else 0.0
        )

        # Predict yield factor
        yield_factor = float(np.maximum(yield_model.predict(x_aligned)[0], 0.0))
        yield_factor = min(yield_factor, 1.0)

        # Predict ratio distribution
        ratio_pred = ratio_model.predict(x_aligned)
        ratio_pred = np.clip(ratio_pred, 0.0, 1.0)
        ratio_pred = normalize_ratios(ratio_pred)
        ratio_pred = apply_zero_history_guardrail(
            ratio_pred, self.template_df, input_species, self.bin_cols
        )
        if max_input_length > 0:
            ratio_pred = max_length_check(ratio_pred, self.bin_cols, max_input_length)

        # Calculate board counts
        total_output = min(total_bfin * yield_factor, total_bfin)
        counts = ratio_pred[0] * total_output
        counts_rounded = round_counts_preserve_total(counts, total_output, total_bfin)

        # Build distribution list
        distribution = []
        for bin_name, qty in zip(self.bin_cols, counts_rounded):
            if qty <= 0:
                continue
            if bin_name == "BIN_OTHER":
                distribution.append(
                    {"grade": "OTHER", "length": "", "width": "", "boards": int(qty)}
                )
                continue
            parts = bin_name.split("_", 3)
            if len(parts) == 4:
                grade, length, width = parts[1], parts[2], parts[3]
            else:
                grade, length, width = "UNK", "", ""
            distribution.append(
                {"grade": grade, "length": length, "width": width, "boards": int(qty)}
            )

        return {
            "manufacturing_order": order_id,
            "total_bfin": round(total_bfin, 2),
            "yield_factor": round(yield_factor, 4),
            "predicted_output": round(total_output, 2),
            "total_boards": int(counts_rounded.sum()),
            "distribution": distribution,
        }

    def _extract_order_metadata(self, df: pd.DataFrame) -> dict:
        """Pre-compute per-order species list and max input length."""
        metadata = {}
        for order_id, group in df.groupby("MANUFACTURINGORDER"):
            species = (
                group["MATERIALSPECIE"].dropna().astype(str).unique().tolist()
                if "MATERIALSPECIE" in group.columns
                else []
            )
            max_length = (
                float(group["TALLYLENGTH"].max())
                if "TALLYLENGTH" in group.columns
                else 0.0
            )
            metadata[str(order_id)] = {
                "species": species,
                "max_length": max_length,
            }
        return metadata

    def _build_distribution(self, counts_rounded: np.ndarray) -> list[dict]:
        """Build distribution list from rounded bin counts."""
        distribution = []
        for bin_name, qty in zip(self.bin_cols, counts_rounded):
            if qty <= 0:
                continue
            if bin_name == "BIN_OTHER":
                distribution.append(
                    {"grade": "OTHER", "length": "", "width": "", "boards": int(qty)}
                )
                continue
            parts = bin_name.split("_", 3)
            if len(parts) == 4:
                grade, length, width = parts[1], parts[2], parts[3]
            else:
                grade, length, width = "UNK", "", ""
            distribution.append(
                {"grade": grade, "length": length, "width": width, "boards": int(qty)}
            )
        return distribution

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        """Vectorized prediction for all orders at once.

        Calls build_features() and model.predict() once for all orders,
        then applies per-order guardrails. Much faster than sequential
        predict_single_order() for large batches.
        """
        ratio_model = self.model_bundle["ratio_model"]
        yield_model = self.model_bundle["yield_model"]

        # Build features for ALL orders in one call
        features = build_features(df)
        aligned, total_bfin_values, order_ids = align_features_batch(
            features, self.template_df
        )

        # Batch predict — one call each instead of N calls
        yield_factors = np.maximum(yield_model.predict(aligned), 0.0)
        yield_factors = np.minimum(yield_factors, 1.0)

        ratio_preds = ratio_model.predict(aligned)
        ratio_preds = np.clip(ratio_preds, 0.0, 1.0)
        ratio_preds = normalize_ratios(ratio_preds)

        # Pre-compute per-order metadata for guardrails
        order_metadata = self._extract_order_metadata(df)

        # Per-order guardrails + distribution (cheap numpy ops)
        results = []
        for i, order_id in enumerate(order_ids):
            meta = order_metadata.get(str(order_id), {})
            ratio_row = ratio_preds[i : i + 1]  # keep 2D shape (1, B)

            ratio_row = apply_zero_history_guardrail(
                ratio_row, self.template_df, meta.get("species", []), self.bin_cols
            )
            max_len = meta.get("max_length", 0.0)
            if max_len > 0:
                ratio_row = max_length_check(ratio_row, self.bin_cols, max_len)

            total_bfin = float(total_bfin_values[i])
            yf = float(yield_factors[i])
            total_output = min(total_bfin * yf, total_bfin)
            counts = ratio_row[0] * total_output
            counts_rounded = round_counts_preserve_total(
                counts, total_output, total_bfin
            )

            results.append(
                {
                    "manufacturing_order": str(order_id),
                    "total_bfin": round(total_bfin, 2),
                    "yield_factor": round(yf, 4),
                    "predicted_output": round(total_output, 2),
                    "total_boards": int(counts_rounded.sum()),
                    "distribution": self._build_distribution(counts_rounded),
                }
            )

        # Free large intermediates for big batches
        if len(order_ids) > 100:
            del aligned, ratio_preds, yield_factors, features
            gc.collect()

        return results

    def predict(self, records: list[dict]) -> dict:
        """Run prediction for one or many manufacturing orders.

        Args:
            records: List of tally record dicts with keys matching TallyRecord schema.

        Returns:
            Dict with num_orders, totals, avg_yield_factor, and per-order results.
        """
        df = pd.DataFrame(records)
        records_received = len(df)

        # Filter to 261 records only
        if "GOODSMOVEMENTTYPE" in df.columns:
            df = df[df["GOODSMOVEMENTTYPE"].astype(str) == "261"].copy()

        if df.empty:
            return {
                "num_orders": 0,
                "total_bfin": 0.0,
                "avg_yield_factor": 0.0,
                "total_predicted_output": 0.0,
                "total_boards": 0,
                "orders": [],
                "records_received": records_received,
                "records_processed": 0,
            }

        if "MANUFACTURINGORDER" not in df.columns:
            df["MANUFACTURINGORDER"] = "ORDER_0001"

        n_orders = df["MANUFACTURINGORDER"].nunique()
        logger.info(
            "Multi-output predict: %d records, %d after 261 filter, %d unique orders",
            records_received, len(df), n_orders,
        )

        # Vectorized batch prediction
        order_results = self.predict_batch(df)

        # Aggregate summary
        total_bfin = sum(r["total_bfin"] for r in order_results)
        total_output = sum(r["predicted_output"] for r in order_results)
        total_boards = sum(r["total_boards"] for r in order_results)
        yf_list = [r["yield_factor"] for r in order_results]
        avg_yield = sum(yf_list) / len(yf_list) if yf_list else 0.0

        logger.info("Completed: %d orders predicted", len(order_results))

        return {
            "num_orders": len(order_results),
            "total_bfin": round(total_bfin, 2),
            "avg_yield_factor": round(avg_yield, 4),
            "total_predicted_output": round(total_output, 2),
            "total_boards": total_boards,
            "orders": order_results,
            "records_received": records_received,
            "records_processed": len(df),
        }
