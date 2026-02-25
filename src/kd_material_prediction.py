"""
KD Material Distribution Prediction Module.

ML-based prediction system that determines KD (Output Material) distribution
from KS (Input Material) properties using:
1. XGBoost regression for yield factor prediction
2. KNN (K-Nearest Neighbors) for KD material distribution prediction

The KNN approach finds the most similar historical manufacturing orders
based on input features (material, species, grade, thickness, plant, etc.)
and averages their actual KD output distributions. This is fully data-driven
and learned from historical 261 (input) & 101 (output) production data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.multi_output_prediction import build_features, normalize_ratios


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def build_kd_training_matrix(
    df_261: pd.DataFrame,
    df_101: pd.DataFrame,
    min_order_threshold: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build training matrix with KD material distribution targets.

    Each row = one manufacturing order.
    Features = input properties (species, thickness, plant, grade, dimensions,
               material code one-hot).
    Targets = one column per KD material (fraction of total output BF, sums to 1).
    """
    order_col = "MANUFACTURINGORDER"

    # --- Clean 261 ---
    df_261 = df_261.copy()
    df_261.columns = df_261.columns.str.upper().str.strip()
    if order_col in df_261.columns:
        df_261[order_col] = df_261[order_col].astype(str).str.strip()
    for col in ["BFIN", "MATERIALTHICKNESS", "TALLYLENGTH", "TALLYWIDTH"]:
        if col in df_261.columns:
            df_261[col] = pd.to_numeric(df_261[col], errors="coerce").fillna(0)
    if "GOODSMOVEMENTTYPE" in df_261.columns:
        df_261 = df_261[df_261["GOODSMOVEMENTTYPE"].astype(str) == "261"].copy()
    if "IS_DELETED" in df_261.columns:
        df_261 = df_261[df_261["IS_DELETED"] != True].copy()  # noqa: E712

    # --- Clean 101 ---
    df_101 = df_101.copy()
    df_101.columns = df_101.columns.str.upper().str.strip()
    if order_col in df_101.columns:
        df_101[order_col] = df_101[order_col].astype(str).str.strip()
    for col in ["BFOUT", "TALLYLENGTH", "TALLYWIDTH"]:
        if col in df_101.columns:
            df_101[col] = pd.to_numeric(df_101[col], errors="coerce").fillna(0)
    if "GOODSMOVEMENTTYPE" in df_101.columns:
        df_101 = df_101[df_101["GOODSMOVEMENTTYPE"].astype(str) == "101"].copy()
    if "IS_DELETED" in df_101.columns:
        df_101 = df_101[df_101["IS_DELETED"] != True].copy()  # noqa: E712

    # --- Build input features (one row per order) ---
    features = build_features(df_261)
    # Flatten any tuple column names from the pivot_table MultiIndex
    features.columns = [
        f"IN_{c[0]}x{c[1]}_VOL" if isinstance(c, tuple) else c
        for c in features.columns
    ]

    # --- Add KS material code as one-hot feature ---
    ks_material_per_order = (
        df_261.groupby(order_col)["MATERIAL"].first().rename("KS_MATERIAL")
    )
    ks_ohe = pd.get_dummies(ks_material_per_order, prefix="KS")
    ks_ohe.index.name = order_col
    features = features.set_index(order_col).join(ks_ohe).reset_index()

    print(f"Built features for {len(features)} orders ({len(ks_ohe.columns)} KS material features)")

    # --- Aggregate 101 per (order, material) ---
    out_per_mat = (
        df_101.groupby([order_col, "MATERIAL"])["BFOUT"]
        .sum()
        .reset_index()
    )
    out_total = (
        df_101.groupby(order_col)["BFOUT"]
        .sum()
        .rename("TOTAL_BFOUT")
    )

    # --- Determine KD column set ---
    mat_order_counts = out_per_mat.groupby("MATERIAL")[order_col].nunique()
    keep_materials = set(
        mat_order_counts[mat_order_counts >= min_order_threshold].index
    )
    print(f"KD materials with >= {min_order_threshold} orders: {len(keep_materials)}")

    # --- Pivot to wide format ---
    out_per_mat["KD_COL"] = out_per_mat["MATERIAL"].apply(
        lambda m: f"KD_{m}" if m in keep_materials else "KD_OTHER"
    )
    kd_wide = (
        out_per_mat.groupby([order_col, "KD_COL"])["BFOUT"]
        .sum()
        .reset_index()
        .pivot(index=order_col, columns="KD_COL", values="BFOUT")
        .fillna(0.0)
    )
    row_totals = kd_wide.sum(axis=1)
    kd_ratios = kd_wide.div(row_totals.replace(0, 1), axis=0)

    kd_cols = sorted(kd_ratios.columns.tolist())
    print(f"KD target columns: {len(kd_cols)} (including KD_OTHER)")

    # --- Compute yield factor ---
    bfin_total = features.set_index(order_col)["TOTAL_BFIN"]
    yield_factor = (out_total / bfin_total).clip(0, 1.0).rename("TOTAL_YIELD_FACTOR")

    # --- Merge features + targets ---
    combined = features.set_index(order_col).join(kd_ratios).join(yield_factor).join(out_total)
    combined = combined.dropna(subset=["TOTAL_BFIN", "TOTAL_YIELD_FACTOR"])
    combined[kd_cols] = combined[kd_cols].fillna(0.0)
    combined = combined.reset_index()

    print(f"Training matrix: {len(combined)} orders, {len(kd_cols)} KD columns")

    return combined, kd_cols


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def apply_input_material_guardrail(
    ratios: np.ndarray,
    material_history: Dict[str, set],
    ks_material: str,
    kd_cols: List[str],
) -> np.ndarray:
    """Zero out KD materials never historically produced from this KS material."""
    if material_history is None or ks_material not in material_history:
        return ratios

    allowed_kd = material_history[ks_material]
    adjusted = ratios.copy()
    for idx, col in enumerate(kd_cols):
        mat_code = col[3:] if col.startswith("KD_") else col
        if col == "KD_OTHER":
            continue
        if mat_code not in allowed_kd:
            adjusted[:, idx] = 0.0
    return normalize_ratios(adjusted)


# ---------------------------------------------------------------------------
# Build material history lookup
# ---------------------------------------------------------------------------

def build_material_history(
    df_261: pd.DataFrame, df_101: pd.DataFrame
) -> Dict[str, set]:
    """Build mapping: {KS_material -> set of KD materials historically produced}."""
    order_col = "MANUFACTURINGORDER"

    input_mat = (
        df_261.groupby(order_col)["MATERIAL"].first().rename("KS_MATERIAL")
    )
    output_mats = (
        df_101.groupby(order_col)["MATERIAL"]
        .apply(set)
        .rename("KD_MATERIALS")
    )

    merged = pd.DataFrame({"KS_MATERIAL": input_mat, "KD_MATERIALS": output_mats}).dropna()

    history: Dict[str, set] = {}
    for _, row in merged.iterrows():
        ks = str(row["KS_MATERIAL"])
        kds = row["KD_MATERIALS"]
        if ks not in history:
            history[ks] = set()
        history[ks].update(str(m) for m in kds)

    print(f"Material history: {len(history)} KS materials mapped to KD outputs")
    return history


# ---------------------------------------------------------------------------
# Prediction (KNN-based)
# ---------------------------------------------------------------------------

def predict_kd_distribution(
    input_data: Dict[str, Any],
    model_bundle: dict,
    template_df: pd.DataFrame,
    kd_cols: List[str],
    material_history: Optional[Dict[str, set]] = None,
    apply_material_guardrail: bool = True,
    df_261_rows: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, dict]:
    """Predict KD material distribution for a single input using KNN.

    Finds the K most similar historical orders and averages their actual
    KD output distributions, weighted by similarity (inverse distance).

    Args:
        input_data: Dict with Input_Plant, Input_Material, Input_Thickness,
            Input_Specie, Input_Grade, Input_Width, Total_Input_BF.
        model_bundle: {'yield_model': XGBRegressor, 'knn_model': NearestNeighbors,
                       'scaler': StandardScaler, 'kd_distributions': np.ndarray,
                       'yield_values': np.ndarray}
        template_df: Training template DataFrame for feature alignment.
        kd_cols: List of KD_* column names.
        material_history: Optional {ks_material -> set(kd_materials)}.
        apply_material_guardrail: Whether to restrict to historically seen outputs.
        df_261_rows: Optional raw 261 DataFrame rows for this order. When
            provided, uses the full rows (with all lengths/widths/grades) to
            build features — matching how training data was processed. When
            None, synthesizes a single row from input_data (manual entry mode).

    Returns:
        (result_df, summary) with Output Material, Distribution %, Expected Output BF.
    """
    yield_model = model_bundle["yield_model"]
    knn_model = model_bundle["knn_model"]
    scaler = model_bundle["scaler"]
    X_train_scaled = model_bundle["X_scaled"]  # (n_train, n_features)
    kd_distributions = model_bundle["kd_distributions"]  # (n_train, n_kd_cols)
    yield_values = model_bundle["yield_values"]  # (n_train,)
    ks_materials = model_bundle.get("ks_materials")  # (n_train,) KS material per row

    MIN_MATERIAL_SAMPLES = 5  # minimum samples to use material-filtered KNN

    ks_material = str(input_data.get("Input_Material", ""))

    if df_261_rows is not None and len(df_261_rows) > 0:
        # --- Full 261 rows mode (file upload) ---
        raw_df = df_261_rows.copy()
        raw_df.columns = raw_df.columns.str.upper().str.strip()
        features = build_features(raw_df)
    else:
        # --- Single synthetic row mode (manual entry) ---
        synthetic = pd.DataFrame(
            [
                {
                    "MANUFACTURINGORDER": "PREDICT_001",
                    "MATERIAL": ks_material,
                    "BFIN": float(input_data.get("Total_Input_BF", 0)),
                    "MATERIALSPECIE": str(input_data.get("Input_Specie", "")),
                    "MATERIALTHICKNESS": float(input_data.get("Input_Thickness", 0)),
                    "PLANT": str(input_data.get("Input_Plant", "")),
                    "TALLYGRADE": str(input_data.get("Input_Grade", "")),
                    "TALLYLENGTH": 96.0,
                    "TALLYWIDTH": float(input_data.get("Input_Width", 0)),
                }
            ]
        )
        features = build_features(synthetic)

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
        col for col in template_df.columns
        if not col.startswith("KD_") and col not in drop_cols
    ]
    aligned = features.reindex(columns=feature_cols, fill_value=0.0)
    total_bfin = float(features["TOTAL_BFIN"].iloc[0])

    # --- Scale features ---
    X_scaled = scaler.transform(aligned.values.astype(np.float64))

    # --- Material-aware KNN: pre-filter to same KS material ---
    n_neighbors = knn_model.n_neighbors
    material_mask = None
    search_mode = "full"

    if ks_materials is not None:
        material_mask = (ks_materials == ks_material)
        n_matching = int(material_mask.sum())

        if n_matching >= MIN_MATERIAL_SAMPLES:
            # Enough orders with this exact material → KNN within material subset
            search_mode = f"material_filtered ({n_matching} orders)"
            mat_indices = np.where(material_mask)[0]
            k = min(n_neighbors, n_matching)

            # Build temporary KNN on just matching orders
            mat_knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            mat_knn.fit(X_train_scaled[mat_indices])
            distances, local_indices = mat_knn.kneighbors(X_scaled)
            distances = distances[0]
            # Map local indices back to global training indices
            indices = mat_indices[local_indices[0]]
        else:
            # Not enough exact matches → fall back to full KNN
            search_mode = f"full (only {n_matching} orders for {ks_material})"
            distances, indices = knn_model.kneighbors(X_scaled)
            distances = distances[0]
            indices = indices[0]
    else:
        distances, indices = knn_model.kneighbors(X_scaled)
        distances = distances[0]
        indices = indices[0]

    # --- Weighted average of neighbor distributions ---
    # Use inverse distance weighting (closer neighbors have more influence)
    weights = 1.0 / (distances + 1e-8)
    weights = weights / weights.sum()

    kd_ratios = np.zeros(len(kd_cols))
    for w, idx in zip(weights, indices):
        kd_ratios += w * kd_distributions[idx]
    kd_ratios = kd_ratios.reshape(1, -1)
    kd_ratios = normalize_ratios(kd_ratios)

    # --- Yield: weighted average of neighbor yields ---
    neighbor_yield = sum(w * yield_values[idx] for w, idx in zip(weights, indices))

    # Also get XGBoost yield prediction and blend
    xgb_yield = float(np.clip(yield_model.predict(aligned)[0], 0.0, 1.0))
    # Blend: 60% KNN yield, 40% XGBoost yield
    yield_factor = float(np.clip(0.6 * neighbor_yield + 0.4 * xgb_yield, 0.0, 1.0))

    # --- Apply material guardrail ---
    if apply_material_guardrail and material_history is not None:
        kd_ratios = apply_input_material_guardrail(
            kd_ratios, material_history, ks_material, kd_cols
        )

    # --- Permanently exclude all 3BKD grade materials ---
    for idx, col in enumerate(kd_cols):
        if "3BKD" in col:
            kd_ratios[:, idx] = 0.0
    kd_ratios = normalize_ratios(kd_ratios)

    # --- Merge NKD material quantities into their KD counterparts ---
    for idx, col in enumerate(kd_cols):
        if col.endswith("NKD"):
            kd_counterpart = col[:-3] + "KD"  # e.g. KD_4ROPRNKD → KD_4ROPRKD
            if kd_counterpart in kd_cols:
                kd_idx = kd_cols.index(kd_counterpart)
                kd_ratios[:, kd_idx] += kd_ratios[:, idx]
            kd_ratios[:, idx] = 0.0
    # No renormalize needed — transfer preserves the total sum

    # --- Optionally remove KD_OTHER and renormalize ---
    exclude_other = kwargs.get("exclude_other", False)
    other_idx = None
    for idx, col in enumerate(kd_cols):
        if col == "KD_OTHER":
            other_idx = idx
            break
    if exclude_other and other_idx is not None:
        kd_ratios[:, other_idx] = 0.0
        kd_ratios = normalize_ratios(kd_ratios)

    # --- Calculate BF per KD material ---
    total_output = total_bfin * yield_factor
    per_kd_bf = kd_ratios[0] * total_output

    # --- Count historical orders per KD column for this KS material ---
    hist_counts = {}
    if ks_materials is not None:
        mat_mask = (ks_materials == ks_material)
        mat_dists = kd_distributions[mat_mask]
        for i, col in enumerate(kd_cols):
            hist_counts[col] = int((mat_dists[:, i] > 0).sum())

    # --- Build results ---
    rows = []
    for col, ratio, bf in zip(kd_cols, kd_ratios[0], per_kd_bf):
        if ratio < 0.005:  # skip entries below 0.5%
            continue
        mat_name = col[3:] if col.startswith("KD_") else col
        if col == "KD_OTHER":
            mat_name = "Other Materials (rare KD combined)"
        rows.append(
            {
                "Output Material": mat_name,
                "Input BF": round(total_bfin, 2),
                "Distribution %": round(float(ratio) * 100, 2),
                "Expected Output BF": round(float(bf), 2),
                "Material Yield %": round(float(bf) / total_bfin * 100, 2) if total_bfin > 0 else 0.0,
                "Historical Orders": hist_counts.get(col, 0),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("Expected Output BF", ascending=False).reset_index(drop=True)
    result_df["Cumulative %"] = result_df["Distribution %"].cumsum().round(2)
    result_df = result_df[
        ["Output Material", "Input BF", "Distribution %", "Cumulative %",
         "Expected Output BF", "Material Yield %", "Historical Orders"]
    ]

    summary = {
        "total_bfin": total_bfin,
        "yield_factor": yield_factor,
        "predicted_yield_pct": round(yield_factor * 100, 2),
        "predicted_output": round(total_output, 2),
        "kd_count": len(result_df),
        "n_neighbors_used": len(indices),
        "avg_neighbor_distance": round(float(distances.mean()), 4),
        "search_mode": search_mode,
    }

    return result_df, summary


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_kd_distribution_model(
    years: List[str] = None,
    min_order_threshold: int = 5,
    save_dir: str = "models/kd_distribution",
    n_neighbors: int = 15,
) -> dict:
    """Train the KD distribution model.

    Stage 1: XGBRegressor for yield factor prediction.
    Stage 2: KNN for KD material distribution (stores training data + index).

    The KNN approach finds similar historical orders and averages their actual
    KD distributions. This works well for sparse multi-output prediction.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    from src.data_preparation import load_raw_csv_data

    if years is None:
        years = ["2024", "2025"]

    print("=" * 70)
    print("TRAINING KD MATERIAL DISTRIBUTION MODEL")
    print("=" * 70)

    # --- Load data ---
    df_261, df_101 = load_raw_csv_data(years=years)

    # --- Build training matrix ---
    training_df, kd_cols = build_kd_training_matrix(
        df_261, df_101, min_order_threshold=min_order_threshold
    )

    # --- Build material history ---
    material_history = build_material_history(df_261, df_101)

    # --- Separate features and targets ---
    drop_cols = {"MANUFACTURINGORDER", "TOTAL_BFOUT", "TOTAL_YIELD_FACTOR"}
    feature_cols = [
        col
        for col in training_df.columns
        if col not in kd_cols and col not in drop_cols
    ]

    X = training_df[feature_cols].fillna(0.0)
    y_yield = training_df["TOTAL_YIELD_FACTOR"].fillna(0.0)
    y_kd = training_df[kd_cols].fillna(0.0)

    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Target KD columns: {len(kd_cols)}")
    print(f"Training samples: {len(X)}")

    # --- Train/test split ---
    X_train, X_test, yy_train, yy_test, ykd_train, ykd_test = train_test_split(
        X, y_yield, y_kd, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Stage 1: Yield model (XGBoost) ---
    print("\nTraining yield model (XGBoost)...")
    yield_model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    yield_model.fit(X_train, yy_train)

    yy_pred = yield_model.predict(X_test)
    yield_metrics = {
        "yield_r2": float(r2_score(yy_test, yy_pred)),
        "yield_mae": float(mean_absolute_error(yy_test, yy_pred)),
        "yield_rmse": float(np.sqrt(mean_squared_error(yy_test, yy_pred))),
    }
    print(f"  Yield R²: {yield_metrics['yield_r2']:.4f}")
    print(f"  Yield MAE: {yield_metrics['yield_mae']:.4f}")

    # --- Stage 2: KNN for KD distribution ---
    print(f"\nBuilding KNN model (K={n_neighbors})...")

    # Scale features for distance computation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float64))

    # Fit KNN on training data
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    knn_model.fit(X_train_scaled)

    # Store training distributions and yields for lookup
    kd_distributions = ykd_train.values.astype(np.float64)
    yield_train_values = yy_train.values.astype(np.float64)

    # --- Evaluate KNN on test set ---
    print("  Evaluating KNN distribution prediction on test set...")
    test_distances, test_indices = knn_model.kneighbors(X_test_scaled)

    knn_preds = np.zeros((len(X_test), len(kd_cols)))
    knn_yield_preds = np.zeros(len(X_test))
    for i in range(len(X_test)):
        weights = 1.0 / (test_distances[i] + 1e-8)
        weights = weights / weights.sum()
        for w, idx in zip(weights, test_indices[i]):
            knn_preds[i] += w * kd_distributions[idx]
            knn_yield_preds[i] += w * yield_train_values[idx]

    knn_preds = normalize_ratios(knn_preds)

    kd_mae = float(mean_absolute_error(ykd_test.values, knn_preds))
    kd_r2 = float(r2_score(ykd_test.values, knn_preds))
    knn_yield_r2 = float(r2_score(yy_test, knn_yield_preds))

    ratio_metrics = {
        "kd_ratio_mae": kd_mae,
        "kd_ratio_r2": kd_r2,
        "knn_yield_r2": knn_yield_r2,
    }
    print(f"  KNN KD Ratio MAE: {kd_mae:.4f}")
    print(f"  KNN KD Ratio R²: {kd_r2:.4f}")
    print(f"  KNN Yield R²: {knn_yield_r2:.4f}")

    # --- Now retrain on ALL data for production use ---
    print("\nRetraining on full dataset for production...")
    scaler_full = StandardScaler()
    X_all_scaled = scaler_full.fit_transform(X.values.astype(np.float64))

    knn_full = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    knn_full.fit(X_all_scaled)

    yield_model_full = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    yield_model_full.fit(X, y_yield)

    # --- Save artifacts ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Per-order KS material for material-aware KNN filtering
    ks_per_order = training_df.set_index("MANUFACTURINGORDER")[
        [c for c in training_df.columns if c.startswith("KS_")]
    ].idxmax(axis=1).str.replace("KS_", "", n=1).values

    model_bundle = {
        "yield_model": yield_model_full,
        "knn_model": knn_full,
        "scaler": scaler_full,
        "X_scaled": X_all_scaled,  # needed for material-filtered KNN
        "kd_distributions": y_kd.values.astype(np.float64),
        "yield_values": y_yield.values.astype(np.float64),
        "ks_materials": ks_per_order,  # KS material per training row
    }
    joblib.dump(model_bundle, save_path / "kd_model_bundle.joblib", compress=3)
    print(f"\nSaved: {save_path / 'kd_model_bundle.joblib'}")

    # Template (full training data for feature alignment + species guardrail)
    template_with_data = training_df[feature_cols + kd_cols].copy()
    joblib.dump(template_with_data, save_path / "kd_training_template.joblib", compress=3)
    print(f"Saved: {save_path / 'kd_training_template.joblib'}")

    # Material history
    joblib.dump(material_history, save_path / "kd_material_history.joblib", compress=3)
    print(f"Saved: {save_path / 'kd_material_history.joblib'}")

    # Metadata
    metrics = {
        **yield_metrics,
        **ratio_metrics,
        "n_kd_columns": len(kd_cols),
        "n_features": len(feature_cols),
        "n_training_samples": len(X),
        "n_test_samples": len(X_test),
        "n_neighbors": n_neighbors,
        "min_order_threshold": min_order_threshold,
        "years": years,
        "kd_columns": kd_cols,
        "trained_at": datetime.now().isoformat(),
    }
    with open(save_path / "model_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {save_path / 'model_metadata.json'}")

    print(f"\n{'=' * 70}")
    print("KD DISTRIBUTION MODEL TRAINING COMPLETE")
    print(f"{'=' * 70}")

    return metrics
