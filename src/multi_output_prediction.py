"""
Multi-Output Yield Prediction Module.

Two-step XGBoost model that predicts:
1. Overall yield factor (what fraction of input becomes output)
2. Output BIN distribution (how output is split across Grade/Length/Width combinations)

Functions copied from multi-output-yield_V2 streamlit_app.py — no logic changes.
Only adaptation: align_features() accepts a template DataFrame instead of a file path.
"""

from __future__ import annotations

import gc
from typing import Optional

import joblib
import numpy as np
import pandas as pd


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
    """Align features to match training template columns.

    Adapted to accept a DataFrame directly instead of reading from file.
    """
    bin_cols = [col for col in template_df.columns if col.startswith("BIN_")]
    drop_cols = {"MANUFACTURINGORDER", "TOTAL_BFOUT", "TOTAL_YIELD_FACTOR"}
    feature_cols = [col for col in template_df.columns if col not in bin_cols]
    feature_cols = [col for col in feature_cols if col not in drop_cols]

    total_bfin = float(features["TOTAL_BFIN"].iloc[0])
    aligned = features.reindex(columns=feature_cols, fill_value=0.0)
    return aligned, total_bfin


def normalize_ratios(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.maximum(y_pred, 0.0)
    denom = np.where(y_pred.sum(axis=1, keepdims=True) == 0.0, 1.0, y_pred.sum(axis=1, keepdims=True))
    return y_pred / denom


def apply_zero_history_guardrail(
    ratios: np.ndarray,
    template_df: pd.DataFrame,
    input_species: list,
    bin_cols: list,
) -> np.ndarray:
    if not input_species:
        return ratios
    species_cols = [f"SPECIE_{val}" for val in input_species if f"SPECIE_{val}" in template_df]
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


def run_prediction_single_order(
    df_order: pd.DataFrame,
    model_bundle: dict,
    template_df: pd.DataFrame,
    bin_cols: list,
) -> tuple:
    """Run prediction for a single manufacturing order."""
    ratio_model = model_bundle["ratio_model"]
    yield_model = model_bundle["yield_model"]

    if df_order.empty:
        return None, None

    # Build features
    features = build_features(df_order)
    x_aligned, total_bfin = align_features(features, template_df)

    # Extract species info for guardrails
    input_species = (
        df_order["MATERIALSPECIE"].dropna().astype(str).unique().tolist()
        if "MATERIALSPECIE" in df_order.columns
        else []
    )
    max_input_length = (
        float(df_order["TALLYLENGTH"].max()) if "TALLYLENGTH" in df_order.columns else 0.0
    )

    # Predict
    yield_factor = float(np.maximum(yield_model.predict(x_aligned)[0], 0.0))
    yield_factor = min(yield_factor, 1.0)

    ratio_pred = ratio_model.predict(x_aligned)
    ratio_pred = np.clip(ratio_pred, 0.0, 1.0)
    ratio_pred = normalize_ratios(ratio_pred)
    ratio_pred = apply_zero_history_guardrail(ratio_pred, template_df, input_species, bin_cols)
    if max_input_length > 0:
        ratio_pred = max_length_check(ratio_pred, bin_cols, max_input_length)

    # Calculate counts
    total_output = min(total_bfin * yield_factor, total_bfin)
    counts = ratio_pred[0] * total_output
    counts_rounded = round_counts_preserve_total(counts, total_output, total_bfin)

    # Build output dataframe
    output = []
    for bin_name, qty in zip(bin_cols, counts_rounded):
        if qty <= 0:
            continue
        if bin_name == "BIN_OTHER":
            output.append({"Grade": "OTHER", "Length": "", "Width": "", "Boards": int(qty)})
            continue
        parts = bin_name.split("_", 3)
        if len(parts) == 4:
            grade, length, width = parts[1], parts[2], parts[3]
        else:
            grade, length, width = "UNK", "", ""
        output.append({"Grade": grade, "Length": length, "Width": width, "Boards": int(qty)})

    result_df = pd.DataFrame(output)

    summary = {
        "total_bfin": total_bfin,
        "yield_factor": yield_factor,
        "predicted_output": total_output,
        "total_boards": int(counts_rounded.sum()),
    }

    return result_df, summary


def run_prediction_all_orders(
    df_input: pd.DataFrame,
    model_bundle: dict,
    template_df: pd.DataFrame,
    progress_bar=None,
) -> tuple:
    """Run prediction for all manufacturing orders in the input data.

    Adapted to accept model_bundle and template_df as parameters
    instead of loading from global paths.
    """
    bin_cols = [col for col in template_df.columns if col.startswith("BIN_")]

    # Filter to 261 records only
    if "GOODSMOVEMENTTYPE" in df_input.columns:
        df_input = df_input[df_input["GOODSMOVEMENTTYPE"].astype(str) == "261"].copy()

    if df_input.empty:
        return None, None

    if "MANUFACTURINGORDER" not in df_input.columns:
        df_input["MANUFACTURINGORDER"] = "ORDER_0001"

    # Get unique orders
    orders = df_input["MANUFACTURINGORDER"].unique()
    num_orders = len(orders)

    all_results = []
    total_bfin = 0.0
    total_output = 0.0
    total_boards = 0
    yield_factors = []

    for i, order_id in enumerate(orders):
        if progress_bar is not None:
            progress_bar.progress((i + 1) / num_orders, text=f"Processing order {i + 1} of {num_orders}")

        order_data = df_input[df_input["MANUFACTURINGORDER"] == order_id].copy()
        result_df, summary = run_prediction_single_order(order_data, model_bundle, template_df, bin_cols)

        if result_df is not None and summary is not None:
            all_results.append(result_df)
            total_bfin += summary["total_bfin"]
            total_output += summary["predicted_output"]
            total_boards += summary["total_boards"]
            yield_factors.append(summary["yield_factor"])

    if not all_results:
        return None, None

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Free memory from intermediate results
    del all_results
    gc.collect()

    # Aggregate by Grade/Length/Width across all orders
    aggregated = (
        combined.groupby(["Grade", "Length", "Width"], as_index=False)["Boards"]
        .sum()
        .sort_values(by="Boards", ascending=False)
        .reset_index(drop=True)
    )

    # Calculate average yield factor
    avg_yield_factor = sum(yield_factors) / len(yield_factors) if yield_factors else 0.0

    summary = {
        "total_bfin": total_bfin,
        "yield_factor": avg_yield_factor,
        "predicted_output": total_output,
        "total_boards": total_boards,
        "num_orders": num_orders,
    }

    return aggregated, summary
