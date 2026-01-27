"""
Prediction Utilities for Material Yield Prediction System

Provides helper functions for the Streamlit app to interact with the
YieldPredictionModel for yield predictions and output material simulation.

SAP Manufacturing Logic:
- 261 = Goods Issue to Order = INPUT (raw materials CONSUMED from stock)
- 101 = Goods Receipt = OUTPUT (finished goods RECEIVED into stock)
- Yield = Total_Output_BF (from 101) / Total_Input_BF (from 261)
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List, Optional, Tuple

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_prediction_artifacts(path: str = None) -> Optional[Dict[str, Any]]:
    """
    Load all prediction model artifacts from disk.

    Returns dict with:
    - model: The trained YieldPredictionModel or best sklearn model
    - classifier: The trained OutputMaterialClassifier (if available)
    - encoders: LabelEncoders for categorical features
    - feature_columns: List of feature column names
    - metrics: Dict of model performance metrics
    - classifier_metrics: Dict of classifier performance metrics
    - test_results: DataFrame with actual vs predicted on test set
    """
    # Use default models directory if not provided
    if path is None:
        path = MODELS_DIR

    try:
        if not os.path.exists(os.path.join(path, "yield_model.joblib")):
            return None

        artifacts = {
            'model': joblib.load(os.path.join(path, "yield_model.joblib")),
            'classifier': None,
            'encoders': {},
            'feature_columns': [],
            'classifier_features': [],
            'metrics': {},
            'classifier_metrics': {},
            'test_results': None
        }

        if os.path.exists(os.path.join(path, "encoders.joblib")):
            artifacts['encoders'] = joblib.load(os.path.join(path, "encoders.joblib"))

        if os.path.exists(os.path.join(path, "feature_columns.joblib")):
            artifacts['feature_columns'] = joblib.load(os.path.join(path, "feature_columns.joblib"))

        if os.path.exists(os.path.join(path, "metrics.joblib")):
            artifacts['metrics'] = joblib.load(os.path.join(path, "metrics.joblib"))

        # Load Output Material Classifier (if available)
        classifier_path = os.path.join(path, "output_classifier.joblib")
        if os.path.exists(classifier_path):
            artifacts['classifier'] = joblib.load(classifier_path)
            print("Loaded output material classifier")

        if os.path.exists(os.path.join(path, "classifier_features.joblib")):
            artifacts['classifier_features'] = joblib.load(os.path.join(path, "classifier_features.joblib"))

        if os.path.exists(os.path.join(path, "classifier_metrics.joblib")):
            artifacts['classifier_metrics'] = joblib.load(os.path.join(path, "classifier_metrics.joblib"))

        # test_results.csv is in data/ folder
        test_results_path = os.path.join(DATA_DIR, "test_results.csv")
        if os.path.exists(test_results_path):
            artifacts['test_results'] = pd.read_csv(test_results_path)

        return artifacts

    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None


def predict_output_material(
    classifier,
    input_data: Dict[str, Any],
    encoders: Dict,
    classifier_features: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Predict the most likely output materials using the classification model.

    Args:
        classifier: Trained output material classifier
        input_data: Dict with input material properties
        encoders: LabelEncoders for decoding
        classifier_features: Feature columns used by classifier
        top_k: Number of top predictions to return

    Returns:
        List of dicts with output_material, probability, rank
    """
    if classifier is None:
        return []

    # Encode categorical fields
    encoded_data = encode_input_data(input_data, encoders)

    # Build feature vector
    X = pd.DataFrame([encoded_data])
    for col in classifier_features:
        if col not in X.columns:
            X[col] = 0
    X = X[classifier_features].fillna(0)

    # Get probabilities
    try:
        proba = classifier.predict_proba(X)[0]
        classes = classifier.classes_

        # Get top-k indices
        top_indices = np.argsort(proba)[::-1][:top_k]

        # Decode material names
        output_encoder = encoders.get('Output_Material')

        results = []
        for rank, idx in enumerate(top_indices, 1):
            encoded_value = classes[idx]

            # Decode to material name
            if output_encoder and hasattr(output_encoder, 'inverse_transform'):
                try:
                    material_name = output_encoder.inverse_transform([encoded_value])[0]
                except:
                    material_name = f"Material_{encoded_value}"
            else:
                material_name = f"Material_{encoded_value}"

            results.append({
                'output_material': material_name,
                'output_material_encoded': int(encoded_value),
                'probability': float(proba[idx]),
                'probability_pct': round(float(proba[idx]) * 100, 2),
                'rank': rank
            })

        return results

    except Exception as e:
        print(f"Error predicting output material: {e}")
        return []


def get_encoder_categories(encoders: Dict) -> Dict[str, List[str]]:
    """
    Extract category lists from LabelEncoders.

    Returns dict mapping column names to their unique categories.
    """
    categories = {}
    for name, encoder in encoders.items():
        if hasattr(encoder, 'classes_'):
            categories[name] = list(encoder.classes_)
    return categories


def encode_input_data(
    input_data: Dict[str, Any],
    encoders: Dict
) -> Dict[str, Any]:
    """
    Encode categorical fields in input data using provided encoders.

    Takes raw input dict with fields like 'Input_Material', 'Input_Specie', etc.
    Returns dict with encoded versions like 'Input_Material_Encoded'.
    """
    encoded = input_data.copy()

    encoding_map = {
        'Input_Material': 'Input_Material_Encoded',
        'Input_Specie': 'Input_Specie_Encoded',
        'Input_Grade': 'Input_Grade_Encoded',
        'Input_Plant': 'Input_Plant_Encoded',
        'Output_Material': 'Output_Material_Encoded',
        'Output_Specie': 'Output_Specie_Encoded',
        'Output_Grade': 'Output_Grade_Encoded'
    }

    for raw_col, encoded_col in encoding_map.items():
        if raw_col in input_data and raw_col in encoders:
            le = encoders[raw_col]
            value = str(input_data[raw_col])
            if value in le.classes_:
                encoded[encoded_col] = le.transform([value])[0]
            else:
                encoded[encoded_col] = 0  # Default for unknown values

    return encoded


def forward_predict(
    model,
    input_data: Dict[str, Any],
    encoders: Dict,
    feature_columns: List[str]
) -> Dict[str, Any]:
    """
    Perform forward prediction: Input BF -> Expected Output BF.

    Args:
        model: Trained sklearn model or YieldPredictionModel
        input_data: Dict with input material properties
        encoders: LabelEncoders for categorical features
        feature_columns: List of feature column names used by model

    Returns:
        Dict with predicted_yield_pct and predicted_output_bf
    """
    # Encode categorical fields
    encoded_data = encode_input_data(input_data, encoders)

    # Build feature vector
    X = pd.DataFrame([encoded_data])

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Select only feature columns in correct order
    X = X[feature_columns].fillna(0)

    # Predict yield percentage
    try:
        predicted_yield = model.predict(X)[0]
        predicted_yield = np.clip(predicted_yield, 0, 150)  # Reasonable bounds
    except Exception as e:
        print(f"Prediction error: {e}")
        predicted_yield = 70.0  # Default fallback

    # Calculate output BF
    input_bf = input_data.get('Total_Input_BF', 0)
    output_bf = input_bf * predicted_yield / 100

    return {
        'predicted_yield_pct': round(predicted_yield, 2),
        'predicted_output_bf': round(output_bf, 2),
        'input_bf': input_bf
    }


def simulate_output_materials(
    model,
    input_materials: List[Dict[str, Any]],
    encoders: Dict,
    feature_columns: List[str],
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulate possible output materials and their predicted yields.

    Given input materials, uses historical data to identify which output
    materials are possible and predicts yield for each combination.

    Args:
        model: Trained sklearn model
        input_materials: List of input material dicts
        encoders: LabelEncoders for categorical features
        feature_columns: Feature columns used by model
        historical_data: DataFrame with historical yield data

    Returns:
        DataFrame with columns:
        - Output_Material
        - Predicted_Yield_Pct
        - Predicted_Output_BF
        - Historical_Yield_Pct
        - Historical_Orders
        - Model_Type
        - Input_Material (concatenated if multiple)
        - Input_BF
    """
    results = []
    total_input_bf = sum(m.get('Total_Input_BF', 0) for m in input_materials)

    # Collect all input material names
    input_mat_names = [m.get('Input_Material', 'Unknown') for m in input_materials]
    input_mat_str = ', '.join(input_mat_names)

    # Find possible output materials from historical data
    if historical_data is None or len(historical_data) == 0:
        return pd.DataFrame()

    # Check if historical_data has the expected columns
    has_input_col = 'Input_Material' in historical_data.columns
    has_output_col = 'Output_Material' in historical_data.columns
    has_yield_col = 'Yield_Percentage' in historical_data.columns

    # Fallback column names (for historical_summary format)
    if not has_yield_col and 'Mean_Yield' in historical_data.columns:
        historical_data = historical_data.rename(columns={'Mean_Yield': 'Yield_Percentage'})
        has_yield_col = True

    if not has_output_col:
        # Cannot simulate without output material info
        return pd.DataFrame()

    # Get unique output materials
    if has_input_col:
        # Filter historical data by input materials
        relevant_hist = historical_data[
            historical_data['Input_Material'].isin(input_mat_names)
        ]
        if len(relevant_hist) == 0:
            relevant_hist = historical_data  # Use all if no match
    else:
        relevant_hist = historical_data

    # Group by output material to get statistics
    output_materials = relevant_hist['Output_Material'].unique()

    for output_mat in output_materials:
        output_hist = relevant_hist[relevant_hist['Output_Material'] == output_mat]

        if len(output_hist) == 0:
            continue

        # Historical statistics
        if has_yield_col:
            hist_yield = output_hist['Yield_Percentage'].mean()
            hist_std = output_hist['Yield_Percentage'].std()
        else:
            hist_yield = 70.0  # Default
            hist_std = 10.0

        hist_count = len(output_hist)

        # Model prediction (average across input materials)
        pred_yields = []
        for inp in input_materials:
            try:
                pred = forward_predict(model, inp, encoders, feature_columns)
                pred_yields.append(pred['predicted_yield_pct'])
            except Exception:
                pred_yields.append(hist_yield)

        model_yield = np.mean(pred_yields) if pred_yields else hist_yield

        # Combine model and historical (weighted average)
        # Weight historical more if we have more data points
        weight = min(hist_count / 100, 0.7)  # Max 70% weight to historical
        final_yield = weight * hist_yield + (1 - weight) * model_yield

        # Calculate actual historical output for reference
        hist_total_output = output_hist['Total_Output_BF'].sum() if 'Total_Output_BF' in output_hist.columns else 0
        hist_total_input = output_hist['Total_Input_BF'].sum() if 'Total_Input_BF' in output_hist.columns else 0
        hist_avg_output = output_hist['Total_Output_BF'].mean() if 'Total_Output_BF' in output_hist.columns else 0

        results.append({
            'Input_Material': input_mat_str,
            'Output_Material': output_mat,
            'Predicted_Yield_Pct': round(final_yield, 2),
            'Model_Yield_Pct': round(model_yield, 2),
            'Historical_Yield_Pct': round(hist_yield, 2),
            'Yield_Std': round(hist_std, 2) if not np.isnan(hist_std) else 0,
            'Historical_Orders': hist_count,
            'Input_BF': total_input_bf,
            'Predicted_Output_BF': round(total_input_bf * final_yield / 100, 2),
            'Historical_Total_Output_BF': round(hist_total_output, 2),
            'Historical_Total_Input_BF': round(hist_total_input, 2),
            'Historical_Avg_Output_BF': round(hist_avg_output, 2),
            'Model_Type': 'Multi-Input' if len(input_materials) > 1 else 'Single-Input'
        })

    if len(results) == 0:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Sort by predicted yield descending - return ALL results, no filtering
    result_df = result_df.sort_values('Predicted_Yield_Pct', ascending=False).reset_index(drop=True)

    return result_df


def get_prediction_summary(
    output_results: pd.DataFrame,
    total_input_bf: float,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a summary dict from prediction results.

    Args:
        output_results: DataFrame from simulate_output_materials
        total_input_bf: Total input board feet
        metrics: Model metrics dict

    Returns:
        Dict with summary statistics for display
    """
    summary = {
        'total_input_bf': total_input_bf,
        'num_outputs': 0,
        'best_output': None,
        'best_yield_pct': 0,
        'best_output_bf': 0,
        'avg_yield_pct': 0,
        'models_evaluated': 0,
        'best_r2': 0,
        'lowest_rmse': float('inf'),
        'multi_input_advantage': False
    }

    if output_results is None or len(output_results) == 0:
        return summary

    summary['num_outputs'] = len(output_results)

    # Best output (highest yield)
    if 'Predicted_Yield_Pct' in output_results.columns:
        best_idx = output_results['Predicted_Yield_Pct'].idxmax()
        best_row = output_results.loc[best_idx]
        summary['best_output'] = best_row.get('Output_Material', 'Unknown')
        summary['best_yield_pct'] = best_row.get('Predicted_Yield_Pct', 0)
        summary['best_output_bf'] = best_row.get('Predicted_Output_BF', 0)
        summary['avg_yield_pct'] = output_results['Predicted_Yield_Pct'].mean()

    # Check for multi-input advantage
    if 'Model_Type' in output_results.columns:
        summary['multi_input_advantage'] = (output_results['Model_Type'] == 'Multi-Input').any()

    # Model metrics
    if metrics:
        summary['models_evaluated'] = len([m for m in metrics.values() if isinstance(m, dict)])

        for name, m in metrics.items():
            if isinstance(m, dict):
                r2 = m.get('test_r2', m.get('R2', 0))
                rmse = m.get('RMSE', float('inf'))

                if r2 > summary['best_r2']:
                    summary['best_r2'] = r2

                if rmse < summary['lowest_rmse']:
                    summary['lowest_rmse'] = rmse

    return summary


def estimate_material_waste(
    input_bf: float,
    predicted_yield_pct: float
) -> Dict[str, float]:
    """
    Estimate material waste from yield prediction.

    Args:
        input_bf: Input board feet
        predicted_yield_pct: Predicted yield percentage

    Returns:
        Dict with waste estimates:
        - output_bf: Expected output board feet
        - waste_bf: Expected waste in board feet
        - waste_pct: Waste percentage
    """
    output_bf = input_bf * predicted_yield_pct / 100
    waste_bf = input_bf - output_bf
    waste_pct = 100 - predicted_yield_pct

    return {
        'input_bf': round(input_bf, 2),
        'output_bf': round(output_bf, 2),
        'waste_bf': round(waste_bf, 2),
        'waste_pct': round(waste_pct, 2)
    }


def reverse_predict(
    model,
    required_output_bf: float,
    input_data: Dict[str, Any],
    encoders: Dict,
    feature_columns: List[str],
    safety_margin_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Reverse prediction: Required Output BF -> Required Input BF.

    Given a target output, calculate how much input material is needed.

    Args:
        model: Trained sklearn model
        required_output_bf: Target output board feet
        input_data: Dict with input material properties
        encoders: LabelEncoders
        feature_columns: Feature columns used by model
        safety_margin_pct: Safety buffer percentage (default 5%)

    Returns:
        Dict with required input quantities
    """
    # Get predicted yield for this input configuration
    pred = forward_predict(model, input_data, encoders, feature_columns)
    predicted_yield = pred['predicted_yield_pct']

    if predicted_yield <= 0:
        predicted_yield = 70.0  # Fallback

    # Calculate required input
    required_input = required_output_bf / (predicted_yield / 100)

    # Add safety margin
    required_with_safety = required_input * (1 + safety_margin_pct / 100)

    return {
        'required_input_bf': round(required_with_safety, 2),
        'required_input_no_safety_bf': round(required_input, 2),
        'target_output_bf': required_output_bf,
        'predicted_yield_pct': predicted_yield,
        'safety_margin_pct': safety_margin_pct
    }


# =============================================================================
# ENHANCED PREDICTION WITH CONFIDENCE & RISK LEVELS
# =============================================================================

def calculate_confidence_level(
    historical_count: int,
    yield_std: float,
    model_r2: float = 0.2
) -> Dict[str, Any]:
    """
    Calculate confidence level based on historical data and model performance.

    Confidence is determined by:
    1. Number of historical orders (more = better)
    2. Yield standard deviation (lower = more consistent)
    3. Model R² score (higher = better predictions)

    Returns:
        Dict with confidence_score, confidence_level, risk_level, and factors
    """
    # Factor 1: Historical data quantity (0-40 points)
    if historical_count >= 100:
        data_score = 40
    elif historical_count >= 50:
        data_score = 30
    elif historical_count >= 20:
        data_score = 20
    elif historical_count >= 5:
        data_score = 10
    else:
        data_score = 5

    # Factor 2: Yield consistency (0-35 points)
    # Lower std = more consistent = higher confidence
    if yield_std <= 5:
        consistency_score = 35
    elif yield_std <= 10:
        consistency_score = 25
    elif yield_std <= 15:
        consistency_score = 15
    elif yield_std <= 20:
        consistency_score = 10
    else:
        consistency_score = 5

    # Factor 3: Model performance (0-25 points)
    if model_r2 >= 0.7:
        model_score = 25
    elif model_r2 >= 0.5:
        model_score = 20
    elif model_r2 >= 0.3:
        model_score = 15
    elif model_r2 >= 0.1:
        model_score = 10
    else:
        model_score = 5

    # Total confidence score (0-100)
    confidence_score = data_score + consistency_score + model_score

    # Determine confidence level
    if confidence_score >= 80:
        confidence_level = "HIGH"
    elif confidence_score >= 60:
        confidence_level = "MEDIUM"
    elif confidence_score >= 40:
        confidence_level = "LOW"
    else:
        confidence_level = "VERY LOW"

    # Risk level (inverse of confidence)
    risk_mapping = {
        "HIGH": "LOW",
        "MEDIUM": "MEDIUM",
        "LOW": "HIGH",
        "VERY LOW": "VERY HIGH"
    }
    risk_level = risk_mapping[confidence_level]

    return {
        'confidence_score': confidence_score,
        'confidence_level': confidence_level,
        'risk_level': risk_level,
        'factors': {
            'data_score': data_score,
            'consistency_score': consistency_score,
            'model_score': model_score,
            'historical_count': historical_count,
            'yield_std': yield_std,
            'model_r2': model_r2
        }
    }


def calculate_yield_range(
    predicted_yield: float,
    historical_yield: float,
    yield_std: float,
    historical_count: int,
    confidence_multiplier: float = 1.96  # 95% confidence interval
) -> Dict[str, float]:
    """
    Calculate yield prediction range (confidence interval).

    Uses historical standard deviation and count to estimate uncertainty.

    Args:
        predicted_yield: Model predicted yield %
        historical_yield: Historical average yield %
        yield_std: Historical yield standard deviation
        historical_count: Number of historical orders
        confidence_multiplier: Z-score for confidence (1.96 = 95%, 1.645 = 90%)

    Returns:
        Dict with yield_min, yield_max, yield_expected, range_width
    """
    # Standard error = std / sqrt(n)
    if historical_count > 1:
        std_error = yield_std / np.sqrt(historical_count)
    else:
        std_error = yield_std if yield_std > 0 else 15.0  # Default uncertainty

    # Margin of error
    margin = confidence_multiplier * std_error

    # Use weighted average of model and historical
    weight = min(historical_count / 100, 0.7)
    expected_yield = weight * historical_yield + (1 - weight) * predicted_yield

    # Calculate range
    yield_min = max(0, expected_yield - margin)
    yield_max = min(150, expected_yield + margin)  # Cap at 150%

    return {
        'yield_min': round(yield_min, 2),
        'yield_max': round(yield_max, 2),
        'yield_expected': round(expected_yield, 2),
        'range_width': round(yield_max - yield_min, 2),
        'margin_of_error': round(margin, 2)
    }


def simulate_output_materials_enhanced(
    model,
    input_materials: List[Dict[str, Any]],
    encoders: Dict,
    feature_columns: List[str],
    historical_data: pd.DataFrame,
    model_metrics: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Enhanced output material simulation with confidence levels and yield ranges.

    This is the improved version that provides:
    1. Yield prediction with confidence intervals
    2. Risk assessment for each output option
    3. Recommendation scores

    Args:
        model: Trained sklearn model
        input_materials: List of input material dicts
        encoders: LabelEncoders for categorical features
        feature_columns: Feature columns used by model
        historical_data: DataFrame with historical yield data
        model_metrics: Dict with model performance metrics (for confidence calc)

    Returns:
        DataFrame with enhanced prediction results including confidence and risk
    """
    results = []
    total_input_bf = sum(m.get('Total_Input_BF', 0) for m in input_materials)

    # Get best model R² for confidence calculation
    best_r2 = 0.2  # Default
    if model_metrics:
        for name, m in model_metrics.items():
            if isinstance(m, dict):
                r2 = m.get('test_r2', m.get('R2', 0))
                if r2 > best_r2:
                    best_r2 = r2

    # Collect all input material names
    input_mat_names = [m.get('Input_Material', 'Unknown') for m in input_materials]
    input_mat_str = ', '.join(input_mat_names)

    # Find possible output materials from historical data
    if historical_data is None or len(historical_data) == 0:
        return pd.DataFrame()

    # Check if historical_data has the expected columns
    has_input_col = 'Input_Material' in historical_data.columns
    has_output_col = 'Output_Material' in historical_data.columns
    has_yield_col = 'Yield_Percentage' in historical_data.columns

    # Fallback column names
    if not has_yield_col and 'Mean_Yield' in historical_data.columns:
        historical_data = historical_data.rename(columns={'Mean_Yield': 'Yield_Percentage'})
        has_yield_col = True

    if not has_output_col:
        return pd.DataFrame()

    # Get unique output materials
    if has_input_col:
        relevant_hist = historical_data[
            historical_data['Input_Material'].isin(input_mat_names)
        ]
        if len(relevant_hist) == 0:
            relevant_hist = historical_data
    else:
        relevant_hist = historical_data

    output_materials = relevant_hist['Output_Material'].unique()

    for output_mat in output_materials:
        output_hist = relevant_hist[relevant_hist['Output_Material'] == output_mat]

        if len(output_hist) == 0:
            continue

        # Historical statistics
        if has_yield_col:
            hist_yield = output_hist['Yield_Percentage'].mean()
            hist_std = output_hist['Yield_Percentage'].std()
            if np.isnan(hist_std):
                hist_std = 15.0
        else:
            hist_yield = 70.0
            hist_std = 15.0

        hist_count = len(output_hist)

        # Model prediction
        pred_yields = []
        for inp in input_materials:
            try:
                pred = forward_predict(model, inp, encoders, feature_columns)
                pred_yields.append(pred['predicted_yield_pct'])
            except Exception:
                pred_yields.append(hist_yield)

        model_yield = np.mean(pred_yields) if pred_yields else hist_yield

        # Calculate yield range (confidence interval)
        yield_range = calculate_yield_range(
            predicted_yield=model_yield,
            historical_yield=hist_yield,
            yield_std=hist_std,
            historical_count=hist_count
        )

        # Calculate confidence level
        confidence = calculate_confidence_level(
            historical_count=hist_count,
            yield_std=hist_std,
            model_r2=best_r2
        )

        # Final predicted yield (expected value)
        final_yield = yield_range['yield_expected']

        # Calculate output BF range
        output_bf_min = total_input_bf * yield_range['yield_min'] / 100
        output_bf_max = total_input_bf * yield_range['yield_max'] / 100
        output_bf_expected = total_input_bf * final_yield / 100

        # Historical totals
        hist_total_output = output_hist['Total_Output_BF'].sum() if 'Total_Output_BF' in output_hist.columns else 0
        hist_total_input = output_hist['Total_Input_BF'].sum() if 'Total_Input_BF' in output_hist.columns else 0
        hist_avg_output = output_hist['Total_Output_BF'].mean() if 'Total_Output_BF' in output_hist.columns else 0

        # Recommendation score (combines yield and confidence)
        # Higher yield + higher confidence = better recommendation
        recommendation_score = (final_yield / 100) * (confidence['confidence_score'] / 100)

        results.append({
            # Basic info
            'Input_Material': input_mat_str,
            'Output_Material': output_mat,
            'Input_BF': total_input_bf,

            # Yield predictions
            'Predicted_Yield_Pct': round(final_yield, 2),
            'Yield_Min_Pct': yield_range['yield_min'],
            'Yield_Max_Pct': yield_range['yield_max'],
            'Yield_Range_Width': yield_range['range_width'],

            # Model vs Historical comparison
            'Model_Yield_Pct': round(model_yield, 2),
            'Historical_Yield_Pct': round(hist_yield, 2),
            'Yield_Std': round(hist_std, 2),

            # Output predictions
            'Predicted_Output_BF': round(output_bf_expected, 2),
            'Output_Min_BF': round(output_bf_min, 2),
            'Output_Max_BF': round(output_bf_max, 2),

            # Confidence & Risk
            'Confidence_Score': confidence['confidence_score'],
            'Confidence_Level': confidence['confidence_level'],
            'Risk_Level': confidence['risk_level'],

            # Historical reference
            'Historical_Orders': hist_count,
            'Historical_Total_Output_BF': round(hist_total_output, 2),
            'Historical_Total_Input_BF': round(hist_total_input, 2),
            'Historical_Avg_Output_BF': round(hist_avg_output, 2),

            # Recommendation
            'Recommendation_Score': round(recommendation_score, 4),
            'Model_Type': 'Multi-Input' if len(input_materials) > 1 else 'Single-Input',

            # Model performance
            'Model_R2': round(best_r2, 4)
        })

    if len(results) == 0:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Sort by recommendation score (best options first)
    result_df = result_df.sort_values('Recommendation_Score', ascending=False).reset_index(drop=True)

    return result_df


def get_top_recommendation(
    output_results: pd.DataFrame,
    total_input_bf: float
) -> Dict[str, Any]:
    """
    Get the top recommended output material with detailed explanation.

    Args:
        output_results: DataFrame from simulate_output_materials_enhanced
        total_input_bf: Total input board feet

    Returns:
        Dict with recommendation details
    """
    if output_results is None or len(output_results) == 0:
        return {
            'has_recommendation': False,
            'message': 'No historical data available for these input materials.'
        }

    # Get top recommendation (already sorted by Recommendation_Score)
    top = output_results.iloc[0]

    # Determine recommendation strength
    if top['Confidence_Level'] == 'HIGH' and top['Predicted_Yield_Pct'] >= 70:
        strength = 'STRONG'
        emoji = '✅'
    elif top['Confidence_Level'] in ['HIGH', 'MEDIUM'] and top['Predicted_Yield_Pct'] >= 60:
        strength = 'MODERATE'
        emoji = '🟡'
    else:
        strength = 'WEAK'
        emoji = '⚠️'

    return {
        'has_recommendation': True,
        'strength': strength,
        'emoji': emoji,
        'output_material': top['Output_Material'],
        'expected_yield': top['Predicted_Yield_Pct'],
        'yield_range': f"{top['Yield_Min_Pct']:.1f}% - {top['Yield_Max_Pct']:.1f}%",
        'expected_output_bf': top['Predicted_Output_BF'],
        'output_range_bf': f"{top['Output_Min_BF']:,.0f} - {top['Output_Max_BF']:,.0f} BF",
        'confidence_level': top['Confidence_Level'],
        'confidence_score': top['Confidence_Score'],
        'risk_level': top['Risk_Level'],
        'historical_orders': top['Historical_Orders'],
        'recommendation_score': top['Recommendation_Score'],
        'yield_std': top.get('Yield_Std', 0),
        'model_r2': top.get('Model_R2', 0.3),
        'historical_yield': top.get('Historical_Yield_Pct', 0),
        'message': f"Based on {top['Historical_Orders']} historical orders, "
                   f"producing {top['Output_Material']} has a {top['Confidence_Level']} confidence level "
                   f"with expected yield of {top['Predicted_Yield_Pct']:.1f}%."
    }


# =============================================================================
# KD MATERIAL LOOKUP - HISTORICAL DISTRIBUTION WITH WASTAGE
# =============================================================================

def get_historical_kd_distribution(
    input_material: str,
    historical_data: pd.DataFrame,
    input_thickness: float = None,
    input_grade: str = None,
    input_species: str = None
) -> List[Dict[str, Any]]:
    """
    Get historical distribution of KD (output) materials WITH GRADE breakdown and YIELD statistics.

    This is a historical lookup showing which output materials and grades have historically
    been produced from this input, along with their average yield percentages.

    Args:
        input_material: Input material code (e.g., "4PO3BKS")
        historical_data: DataFrame with joined 261+101 data (must have Input_Material, Output_Material, Output_Grade, Yield_Percentage columns)
        input_thickness: Optional filter by thickness
        input_grade: Optional filter by grade
        input_species: Optional filter by species

    Returns:
        List of dicts sorted by Output_Material then count descending:
        [
            {'Output_Material': '4PO3BKD', 'Output_Grade': '3B', 'Order_Count': 39, 'Percentage': 34.0, 'Avg_Yield': 92.5, 'Yield_Std': 2.3},
            {'Output_Material': '4PO3BKD', 'Output_Grade': 'PR', 'Order_Count': 1, 'Percentage': 0.9, 'Avg_Yield': 88.1, 'Yield_Std': 0.0},
            ...
        ]
        Returns empty list if no historical data found.
    """
    if historical_data is None or len(historical_data) == 0:
        return []

    # Check required columns
    if 'Input_Material' not in historical_data.columns or 'Output_Material' not in historical_data.columns:
        return []

    # Filter by input material (exact match)
    filtered = historical_data[historical_data['Input_Material'] == input_material].copy()

    # Apply optional filters
    if input_thickness is not None and 'Input_Thickness' in filtered.columns:
        filtered = filtered[filtered['Input_Thickness'] == input_thickness]

    if input_grade is not None and 'Input_Grade' in filtered.columns:
        filtered = filtered[filtered['Input_Grade'] == input_grade]

    if input_species is not None and 'Input_Specie' in filtered.columns:
        filtered = filtered[filtered['Input_Specie'] == input_species]

    if len(filtered) == 0:
        return []

    # Check if Output_Grade and Yield_Percentage columns exist
    has_output_grade = 'Output_Grade' in filtered.columns
    has_yield = 'Yield_Percentage' in filtered.columns

    # Group columns
    group_cols = ['Output_Material']
    if has_output_grade:
        group_cols.append('Output_Grade')

    # Aggregate: count, and yield statistics if available
    agg_dict = {'Output_Material': 'count'}  # This gives us Order_Count

    if has_yield:
        # Calculate yield statistics per group
        yield_stats = filtered.groupby(group_cols).agg({
            'Yield_Percentage': ['mean', 'std', 'count']
        }).reset_index()

        # Flatten column names
        yield_stats.columns = group_cols + ['Avg_Yield', 'Yield_Std', 'Order_Count']

        # Fill NaN std with 0 (for groups with only 1 record)
        yield_stats['Yield_Std'] = yield_stats['Yield_Std'].fillna(0)

        # Round values
        yield_stats['Avg_Yield'] = yield_stats['Avg_Yield'].round(2)
        yield_stats['Yield_Std'] = yield_stats['Yield_Std'].round(2)

        output_counts = yield_stats
    else:
        # No yield data - just count
        output_counts = filtered.groupby(group_cols).size().reset_index(name='Order_Count')
        output_counts['Avg_Yield'] = 0.0
        output_counts['Yield_Std'] = 0.0

    # Add Output_Grade column if not present
    if not has_output_grade:
        output_counts['Output_Grade'] = 'N/A'

    # Calculate percentages
    total_orders = output_counts['Order_Count'].sum()
    output_counts['Percentage'] = (output_counts['Order_Count'] / total_orders * 100).round(2)

    # Sort by Output_Material first, then by Order_Count descending within each material
    output_counts = output_counts.sort_values(
        ['Output_Material', 'Order_Count'],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Ensure column order
    cols = ['Output_Material', 'Output_Grade', 'Order_Count', 'Percentage', 'Avg_Yield', 'Yield_Std']
    output_counts = output_counts[[c for c in cols if c in output_counts.columns]]

    # Convert to list of dicts
    results = output_counts.to_dict('records')

    return results


def calculate_kd_output_with_wastage(
    input_bf: float,
    kd_distribution: List[Dict[str, Any]],
    wastage_pct: float = 9.0
) -> Dict[str, Any]:
    """
    Apply wastage and distribute expected output BF across KD materials and grades.

    Args:
        input_bf: Input board feet (e.g., 10000)
        kd_distribution: List from get_historical_kd_distribution() (includes Output_Grade)
        wastage_pct: Wastage percentage (default 9.0)

    Returns:
        Dict with:
        - 'input_bf': Original input BF
        - 'wastage_pct': Wastage percentage applied
        - 'wastage_bf': BF lost to wastage
        - 'available_bf': BF available after wastage
        - 'kd_outputs': List of dicts with Expected_Output_BF added (includes Output_Grade)
        - 'total_expected_bf': Sum of all expected outputs (should equal available_bf)

    Example:
        Input: 10,000 BF, 9% wastage
        Distribution: [{'Output_Material': '4PO3BKD', 'Output_Grade': '1C', 'Percentage': 34.0}, ...]
        Output: available_bf = 9,100, 4PO3BKD-1C gets 3,094 BF (34% of 9,100)
    """
    if not kd_distribution:
        return {
            'input_bf': input_bf,
            'wastage_pct': wastage_pct,
            'wastage_bf': 0,
            'available_bf': 0,
            'kd_outputs': [],
            'total_expected_bf': 0
        }

    # Calculate available BF after wastage
    wastage_bf = input_bf * (wastage_pct / 100)
    available_bf = input_bf - wastage_bf

    # Distribute available BF across KD materials + grades by percentage
    kd_outputs = []
    total_expected = 0

    for kd in kd_distribution:
        expected_bf = available_bf * (kd['Percentage'] / 100)
        kd_output = {
            'Output_Material': kd['Output_Material'],
            'Output_Grade': kd.get('Output_Grade', 'N/A'),
            'Order_Count': kd['Order_Count'],
            'Percentage': kd['Percentage'],
            'Avg_Yield': kd.get('Avg_Yield', 0.0),  # Historical average yield %
            'Yield_Std': kd.get('Yield_Std', 0.0),  # Yield standard deviation
            'Expected_Output_BF': round(expected_bf, 2)
        }
        kd_outputs.append(kd_output)
        total_expected += expected_bf

    return {
        'input_bf': input_bf,
        'wastage_pct': wastage_pct,
        'wastage_bf': round(wastage_bf, 2),
        'available_bf': round(available_bf, 2),
        'kd_outputs': kd_outputs,
        'total_expected_bf': round(total_expected, 2)
    }
