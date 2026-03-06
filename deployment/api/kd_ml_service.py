"""KD Material Distribution Service — ML-based (KNN + XGBoost).

Two-stage prediction:
1. XGBoost: Predicts overall yield factor
2. KNN: Finds K nearest historical orders, averages their actual KD distributions

Self-contained — prediction functions copied from src/ so the API deployment
has no dependency on the src/ package.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ─── Prediction helper functions (copied from multi_output_service.py) ────

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


def normalize_ratios(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.maximum(y_pred, 0.0)
    denom = np.where(
        y_pred.sum(axis=1, keepdims=True) == 0.0,
        1.0,
        y_pred.sum(axis=1, keepdims=True),
    )
    return y_pred / denom


def apply_input_material_guardrail(
    ratios: np.ndarray,
    material_history: Dict,
    ks_material: str,
    kd_cols: List[str],
    plant: Optional[str] = None,
) -> np.ndarray:
    """Zero out KD materials never historically produced from this KS material.

    Looks up (ks_material, plant) first for plant-specific filtering,
    then falls back to ks_material alone (global) if no plant-specific entry.
    """
    if material_history is None:
        return ratios

    allowed_kd = None
    if plant and (ks_material, plant) in material_history:
        allowed_kd = material_history[(ks_material, plant)]
    elif ks_material in material_history:
        allowed_kd = material_history[ks_material]

    if allowed_kd is None:
        return ratios

    adjusted = ratios.copy()
    for idx, col in enumerate(kd_cols):
        mat_code = col[3:] if col.startswith("KD_") else col
        if col == "KD_OTHER":
            continue
        if mat_code not in allowed_kd:
            adjusted[:, idx] = 0.0
    return normalize_ratios(adjusted)


# ─── Service class ─────────────────────────────────────────────────────

MIN_MATERIAL_SAMPLES = 5


class KDMLDistributionService:
    """Loads the KNN + XGBoost KD distribution model and exposes prediction."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        warnings.filterwarnings("ignore", category=UserWarning)

        self.model_bundle = joblib.load(model_dir / "kd_model_bundle.joblib")
        self.template_df = joblib.load(model_dir / "kd_training_template.joblib")
        self.kd_cols = [c for c in self.template_df.columns if c.startswith("KD_")]

        history_path = model_dir / "kd_material_history.joblib"
        self.material_history = (
            joblib.load(history_path) if history_path.exists() else None
        )

    def predict_order(
        self,
        order_id: str,
        df_261_rows: pd.DataFrame,
        ks_material: str,
        apply_guardrail: bool = True,
        exclude_other: bool = False,
    ) -> Dict[str, Any]:
        """Predict KD distribution for one manufacturing order.

        All tally rows are processed together via build_features(), matching
        exactly how the Streamlit CSV upload works.

        Args:
            order_id: Manufacturing order identifier.
            df_261_rows: DataFrame of ALL 261 tally rows for this order.
            ks_material: KS input material code (first material in order).
            apply_guardrail: Restrict to historically seen KD outputs.
            exclude_other: Redistribute KD_OTHER across named materials.

        Returns:
            Dict with summary fields and distribution list.
        """
        yield_model = self.model_bundle["yield_model"]
        knn_model = self.model_bundle["knn_model"]
        scaler = self.model_bundle["scaler"]
        X_train_scaled = self.model_bundle["X_scaled"]
        kd_distributions = self.model_bundle["kd_distributions"]
        yield_values = self.model_bundle["yield_values"]
        ks_materials = self.model_bundle.get("ks_materials")

        num_tally_rows = len(df_261_rows)

        # --- Build features from ALL rows (same as Streamlit) ---
        features = build_features(df_261_rows)

        # --- Flatten tuple columns from pivot_table MultiIndex ---
        features.columns = [
            f"IN_{c[0]}x{c[1]}_VOL" if isinstance(c, tuple) else c
            for c in features.columns
        ]

        # Add KS material one-hot
        ks_col = f"KS_{ks_material}"
        features[ks_col] = 1

        # --- Align to training feature columns ---
        drop_cols = {"MANUFACTURINGORDER", "TOTAL_BFOUT", "TOTAL_YIELD_FACTOR"}
        feature_cols = [
            col for col in self.template_df.columns
            if not col.startswith("KD_") and col not in drop_cols
        ]
        aligned = features.reindex(columns=feature_cols, fill_value=0.0)
        total_bfin = float(features["TOTAL_BFIN"].iloc[0])

        # --- Scale features ---
        X_scaled = scaler.transform(aligned.values.astype(np.float64))

        # --- Material-aware KNN ---
        n_neighbors = knn_model.n_neighbors
        search_mode = "full"

        if ks_materials is not None:
            material_mask = (ks_materials == ks_material)
            n_matching = int(material_mask.sum())

            if n_matching >= MIN_MATERIAL_SAMPLES:
                search_mode = f"material_filtered ({n_matching} orders)"
                mat_indices = np.where(material_mask)[0]
                k = min(n_neighbors, n_matching)

                mat_knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
                mat_knn.fit(X_train_scaled[mat_indices])
                distances, local_indices = mat_knn.kneighbors(X_scaled)
                distances = distances[0]
                indices = mat_indices[local_indices[0]]
            else:
                search_mode = f"full (only {n_matching} orders for {ks_material})"
                distances, indices = knn_model.kneighbors(X_scaled)
                distances = distances[0]
                indices = indices[0]
        else:
            distances, indices = knn_model.kneighbors(X_scaled)
            distances = distances[0]
            indices = indices[0]

        # --- Weighted average of neighbor distributions ---
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum()

        kd_ratios = np.zeros(len(self.kd_cols))
        for w, idx in zip(weights, indices):
            kd_ratios += w * kd_distributions[idx]
        kd_ratios = kd_ratios.reshape(1, -1)
        kd_ratios = normalize_ratios(kd_ratios)

        # --- Yield: blend KNN + XGBoost ---
        neighbor_yield = sum(w * yield_values[idx] for w, idx in zip(weights, indices))
        xgb_yield = float(np.clip(yield_model.predict(aligned)[0], 0.0, 1.0))
        yield_factor = float(np.clip(0.6 * neighbor_yield + 0.4 * xgb_yield, 0.0, 1.0))

        # --- Apply material guardrail ---
        if apply_guardrail and self.material_history is not None:
            plant = str(df_261_rows["PLANT"].iloc[0]) if "PLANT" in df_261_rows.columns else None
            kd_ratios = apply_input_material_guardrail(
                kd_ratios, self.material_history, ks_material, self.kd_cols, plant=plant
            )

        # --- Permanently exclude all 3BKD grade materials ---
        for idx, col in enumerate(self.kd_cols):
            if "3BKD" in col:
                kd_ratios[:, idx] = 0.0
        kd_ratios = normalize_ratios(kd_ratios)

        # --- Merge NKD material quantities into their KD counterparts ---
        for idx, col in enumerate(self.kd_cols):
            if col.endswith("NKD"):
                kd_counterpart = col[:-3] + "KD"
                if kd_counterpart in self.kd_cols:
                    kd_idx = self.kd_cols.index(kd_counterpart)
                    kd_ratios[:, kd_idx] += kd_ratios[:, idx]
                kd_ratios[:, idx] = 0.0

        # --- Merge S2/S2E1/S2E2 materials into parent (strip S2 + truncate after KD) ---
        s2_orphans = {}
        for idx, col in enumerate(self.kd_cols):
            mat = col[3:] if col.startswith("KD_") else col
            parent = mat
            for pattern in ["S2E1", "S2E2", "S2"]:
                if pattern in parent:
                    parent = parent.replace(pattern, "", 1)
                    break
            if parent == mat:
                continue
            kd_pos = parent.find("KD")
            if kd_pos >= 0:
                parent = parent[:kd_pos + 2]
            parent_col = f"KD_{parent}" if col.startswith("KD_") else parent
            if parent_col in self.kd_cols:
                parent_idx = self.kd_cols.index(parent_col)
                kd_ratios[:, parent_idx] += kd_ratios[:, idx]
            else:
                s2_orphans[parent] = s2_orphans.get(parent, 0) + float(kd_ratios[0, idx])
            kd_ratios[:, idx] = 0.0

        # --- Optionally remove KD_OTHER ---
        if exclude_other:
            for idx, col in enumerate(self.kd_cols):
                if col == "KD_OTHER":
                    kd_ratios[:, idx] = 0.0
                    break
            kd_ratios = normalize_ratios(kd_ratios)

        # --- Calculate BF per KD material ---
        total_output = total_bfin * yield_factor
        per_kd_bf = kd_ratios[0] * total_output

        # --- Count historical orders per KD column ---
        hist_counts = {}
        if ks_materials is not None:
            mat_mask = (ks_materials == ks_material)
            mat_dists = kd_distributions[mat_mask]
            for i, col in enumerate(self.kd_cols):
                hist_counts[col] = int((mat_dists[:, i] > 0).sum())

        # --- Build distribution list ---
        rows = []
        for col, ratio, bf in zip(self.kd_cols, kd_ratios[0], per_kd_bf):
            if ratio < 0.005:
                continue
            mat_name = col[3:] if col.startswith("KD_") else col
            if col == "KD_OTHER":
                mat_name = "Other Materials (rare KD combined)"
            rows.append({
                "material": mat_name,
                "bfout": round(float(bf), 2),
            })

        # --- Add orphaned S2 parent materials not in training columns ---
        for parent_name, orphan_ratio in s2_orphans.items():
            if orphan_ratio < 0.005:
                continue
            rows.append({
                "material": parent_name,
                "bfout": round(float(orphan_ratio * total_output), 2),
            })

        # Sort by bfout descending
        rows.sort(key=lambda r: r["bfout"], reverse=True)

        return {
            "manufacturing_order": order_id,
            "material": ks_material,
            "total_bfin": round(total_bfin, 2),
            "num_tally_rows": num_tally_rows,
            "predicted_yield_pct": round(yield_factor * 100, 2),
            "predicted_output_bf": round(total_output, 2),
            "search_mode": search_mode,
            "n_neighbors_used": len(indices),
            "avg_neighbor_distance": round(float(distances.mean()), 4),
            "kd_count": len(rows),
            "distribution": rows,
        }

    def predict(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group items by manufacturing_order and predict per order.

        Each group of items sharing the same manufacturing_order is combined
        into a single DataFrame and processed together — matching how the
        Streamlit CSV upload works.

        Returns:
            Dict with per-order results, counts, and errors.
        """
        # Build full DataFrame from all items
        records = []
        for item in items:
            records.append({
                "MANUFACTURINGORDER": str(item.get("manufacturing_order", "ORDER_0001")),
                "MATERIAL": str(item["material"]),
                "BFIN": float(item["bfin"]),
                "MATERIALSPECIE": str(item["species"]),
                "MATERIALTHICKNESS": float(item["thickness"]),
                "PLANT": str(item["plant"]),
                "TALLYGRADE": str(item["tallygrade"]),
                "TALLYLENGTH": float(item["tallylength"]),
                "TALLYWIDTH": float(item["tallywidth"]),
            })
        df = pd.DataFrame(records)

        # Group by manufacturing order
        order_ids = df["MANUFACTURINGORDER"].unique()
        results = []
        errors = []

        for order_id in order_ids:
            try:
                order_rows = df[df["MANUFACTURINGORDER"] == order_id].copy()
                ks_material = str(order_rows["MATERIAL"].iloc[0])

                result = self.predict_order(
                    order_id=str(order_id),
                    df_261_rows=order_rows,
                    ks_material=ks_material,
                )
                results.append(result)
            except Exception as e:
                logger.warning("KD ML prediction failed for order %s: %s", order_id, e)
                errors.append({"manufacturing_order": str(order_id), "error": str(e)})

        return {
            "results": results,
            "num_orders": len(order_ids),
            "total_records": len(items),
            "successful": len(results),
            "failed": len(errors),
            "errors": errors,
        }
