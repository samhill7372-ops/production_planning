"""
Test Sample Data with Existing Model

Tests the prediction model using sample_261.csv and sample_101.csv files.
Outputs predictions vs actual for debugging yield and output material issues.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from src.prediction_utils import (
    load_prediction_artifacts,
    forward_predict,
    predict_output_material,
    get_material_level_forward_prediction,
    encode_input_data,
    get_historical_kd_distribution,
    calculate_kd_output_with_wastage
)
from src.data_preparation import load_csv_files

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_sample_data():
    """Load the sample 261 and 101 CSV files."""
    sample_261_path = os.path.join(DATA_DIR, "sample_261.csv")
    sample_101_path = os.path.join(DATA_DIR, "sample_101.csv")

    if not os.path.exists(sample_261_path):
        print(f"ERROR: sample_261.csv not found at {sample_261_path}")
        return None, None

    if not os.path.exists(sample_101_path):
        print(f"ERROR: sample_101.csv not found at {sample_101_path}")
        return None, None

    df_261 = pd.read_csv(sample_261_path)
    df_101 = pd.read_csv(sample_101_path)

    print(f"Loaded sample_261.csv: {len(df_261)} rows")
    print(f"Loaded sample_101.csv: {len(df_101)} rows")

    return df_261, df_101


def analyze_sample_data(df_261, df_101):
    """Analyze the sample data - show summary statistics."""
    print("\n" + "="*70)
    print("SAMPLE DATA ANALYSIS")
    print("="*70)

    # 261 Analysis
    print("\n--- 261 (INPUT) Data Summary ---")
    print(f"Unique Manufacture Orders: {df_261['MANUFACTURINGORDER'].nunique()}")
    print(f"Unique Materials: {df_261['MATERIAL'].nunique()}")
    print(f"Unique Species: {df_261['MATERIALSPECIE'].nunique()}")
    print(f"Unique Plants: {df_261['PLANT'].nunique()}")
    print(f"Total Input BF: {df_261['BFIN'].sum():,.0f}")

    print("\nMaterials breakdown:")
    mat_summary = df_261.groupby(['MANUFACTURINGORDER', 'MATERIAL', 'MATERIALSPECIE', 'PLANT']).agg({
        'BFIN': 'sum'
    }).reset_index()
    print(mat_summary.to_string(index=False))

    # 101 Analysis
    print("\n--- 101 (OUTPUT) Data Summary ---")
    print(f"Unique Manufacture Orders: {df_101['MANUFACTURINGORDER'].nunique()}")
    print(f"Unique Materials: {df_101['MATERIAL'].nunique()}")
    print(f"Total Output BF: {df_101['BFOUT'].sum():,.0f}")

    # Calculate actual yields per order
    print("\n--- Actual Yields per Order ---")
    input_by_order = df_261.groupby('MANUFACTURINGORDER')['BFIN'].sum()
    output_by_order = df_101.groupby('MANUFACTURINGORDER')['BFOUT'].sum()

    yields = pd.DataFrame({
        'Input_BF': input_by_order,
        'Output_BF': output_by_order
    }).fillna(0)
    yields['Actual_Yield_Pct'] = (yields['Output_BF'] / yields['Input_BF'] * 100).round(2)

    print(yields.to_string())
    print(f"\nOverall Actual Yield: {yields['Output_BF'].sum() / yields['Input_BF'].sum() * 100:.2f}%")

    return yields


def test_predictions(df_261, df_101, artifacts):
    """Test model predictions against sample data."""
    print("\n" + "="*70)
    print("MODEL PREDICTIONS vs ACTUAL")
    print("="*70)

    if artifacts is None:
        print("ERROR: No model artifacts loaded!")
        return

    model = artifacts['model']
    encoders = artifacts['encoders']
    feature_columns = artifacts['feature_columns']
    classifier = artifacts.get('classifier')
    classifier_features = artifacts.get('classifier_features', [])

    print(f"\nModel loaded: {type(model).__name__}")
    print(f"Feature columns: {feature_columns}")
    print(f"Classifier available: {classifier is not None}")

    # Get unique orders
    orders = df_261['MANUFACTURINGORDER'].unique()

    results = []

    for order in orders:
        order_261 = df_261[df_261['MANUFACTURINGORDER'] == order]
        order_101 = df_101[df_101['MANUFACTURINGORDER'] == order]

        # Aggregate input
        total_input_bf = order_261['BFIN'].sum()
        total_output_bf = order_101['BFOUT'].sum()
        actual_yield = (total_output_bf / total_input_bf * 100) if total_input_bf > 0 else 0

        # Get dominant input characteristics
        input_material = order_261['MATERIAL'].mode().iloc[0] if len(order_261) > 0 else 'Unknown'
        input_specie = order_261['MATERIALSPECIE'].mode().iloc[0] if len(order_261) > 0 else 'Unknown'
        input_plant = order_261['PLANT'].mode().iloc[0] if len(order_261) > 0 else 'Unknown'
        input_thickness = order_261['MATERIALTHICKNESS'].mean() if len(order_261) > 0 else 0
        input_grade = order_261['TALLYGRADE'].mode().iloc[0] if len(order_261) > 0 else 'Unknown'
        input_length = order_261['TALLYLENGTH'].mean() if len(order_261) > 0 else 0
        input_width = order_261['TALLYWIDTH'].mean() if len(order_261) > 0 else 0

        # Build input dict for prediction
        input_data = {
            'Input_Plant': str(input_plant),
            'Input_Material': str(input_material),
            'Input_Thickness': float(input_thickness),
            'Input_Specie': str(input_specie),
            'Input_Grade': str(input_grade),
            'Input_Length': float(input_length),
            'Input_Width': float(input_width),
            'Total_Input_BF': float(total_input_bf)
        }

        # Make yield prediction
        try:
            pred_result = forward_predict(model, input_data, encoders, feature_columns)
            predicted_yield = pred_result['predicted_yield_pct']
            predicted_output_bf = pred_result['predicted_output_bf']
        except Exception as e:
            print(f"  Prediction error for {order}: {e}")
            predicted_yield = 0
            predicted_output_bf = 0

        # Test output material classifier
        predicted_materials = []
        if classifier is not None and len(classifier_features) > 0:
            try:
                mat_predictions = predict_output_material(
                    classifier, input_data, encoders, classifier_features, top_k=3
                )
                predicted_materials = [p['output_material'] for p in mat_predictions]
            except Exception as e:
                print(f"  Classifier error for {order}: {e}")

        # Actual output materials
        actual_materials = order_101['MATERIAL'].unique().tolist() if len(order_101) > 0 else []

        yield_error = predicted_yield - actual_yield

        results.append({
            'Order': order,
            'Input_Material': input_material,
            'Input_Specie': input_specie,
            'Input_Plant': input_plant,
            'Input_BF': total_input_bf,
            'Actual_Output_BF': total_output_bf,
            'Predicted_Output_BF': predicted_output_bf,
            'Actual_Yield_%': round(actual_yield, 2),
            'Predicted_Yield_%': round(predicted_yield, 2),
            'Yield_Error_%': round(yield_error, 2),
            'Actual_Materials': actual_materials,
            'Predicted_Materials': predicted_materials
        })

    # Display results
    print("\n--- Prediction Results ---\n")

    for r in results:
        print(f"Order: {r['Order']}")
        print(f"  Input: {r['Input_Material']} ({r['Input_Specie']}) @ {r['Input_Plant']}")
        print(f"  Input BF: {r['Input_BF']:,.0f}")
        print(f"  Actual Yield: {r['Actual_Yield_%']:.2f}%  |  Predicted Yield: {r['Predicted_Yield_%']:.2f}%  |  Error: {r['Yield_Error_%']:+.2f}%")
        print(f"  Actual Output BF: {r['Actual_Output_BF']:,.0f}  |  Predicted Output BF: {r['Predicted_Output_BF']:,.0f}")
        print(f"  Actual Output Materials: {r['Actual_Materials'][:5]}")  # Show first 5
        print(f"  Predicted Materials: {r['Predicted_Materials']}")
        print()

    # Summary statistics
    print("\n--- SUMMARY ---")
    df_results = pd.DataFrame(results)

    mae = df_results['Yield_Error_%'].abs().mean()
    rmse = np.sqrt((df_results['Yield_Error_%'] ** 2).mean())

    print(f"Mean Absolute Error (Yield): {mae:.2f}%")
    print(f"Root Mean Square Error (Yield): {rmse:.2f}%")
    print(f"Average Predicted Yield: {df_results['Predicted_Yield_%'].mean():.2f}%")
    print(f"Average Actual Yield: {df_results['Actual_Yield_%'].mean():.2f}%")

    # Check material prediction accuracy
    material_matches = 0
    total_predictions = 0
    for r in results:
        if r['Predicted_Materials']:
            total_predictions += 1
            # Check if any predicted material is in actual
            if any(pm in r['Actual_Materials'] for pm in r['Predicted_Materials']):
                material_matches += 1

    if total_predictions > 0:
        material_accuracy = material_matches / total_predictions * 100
        print(f"Material Prediction Accuracy (top-3 contains actual): {material_accuracy:.1f}%")

    return df_results


def test_material_level_prediction(df_261, df_101):
    """Test material-level forward prediction."""
    print("\n" + "="*70)
    print("MATERIAL-LEVEL FORWARD PREDICTION TEST")
    print("="*70)

    # Get unique KS materials and plants
    unique_combos = df_261.groupby(['MATERIAL', 'PLANT']).agg({
        'BFIN': 'sum'
    }).reset_index()

    print(f"\nTesting {len(unique_combos)} unique Material+Plant combinations:\n")

    for _, row in unique_combos.iterrows():
        ks_material = row['MATERIAL']
        plant = row['PLANT']
        input_bf = 1000  # Test with 1000 BF

        result = get_material_level_forward_prediction(
            ks_material=ks_material,
            plant=plant,
            input_bf=input_bf,
            df_261_raw=df_261,
            df_101_raw=df_101,
            min_order_count=0  # Lower threshold for sample data
        )

        if 'error' in result:
            print(f"  {ks_material} @ {plant}: ERROR - {result['error']}")
        else:
            print(f"  {ks_material} @ {plant}:")
            print(f"    Historical Yield: {result['historical_yield_pct']:.2f}%")
            print(f"    Predicted Output for {input_bf} BF input: {result['predicted_output_bf']:.2f} BF")
            print(f"    KD Materials ({result['kd_materials_count']}):")
            for kd in result['kd_distribution'][:3]:  # Show top 3
                print(f"      - {kd['KD_Material']}: {kd['Contribution_Pct']:.1f}% ({kd['Expected_BF_Output']:.0f} BF)")
        print()


def main():
    """Main test function."""
    print("="*70)
    print("SAMPLE DATA MODEL TEST")
    print("="*70)

    # Load sample data
    df_261, df_101 = load_sample_data()
    if df_261 is None or df_101 is None:
        return

    # Analyze sample data
    yields = analyze_sample_data(df_261, df_101)

    # Load model artifacts (try 2025 first, then 2024)
    artifacts = None
    for year in ['2025', '2024']:
        model_path = os.path.join(MODELS_DIR, year)
        if os.path.exists(os.path.join(model_path, "yield_model.joblib")):
            print(f"\nLoading model from: {model_path}")
            artifacts = load_prediction_artifacts(model_path)
            if artifacts:
                print(f"Model loaded successfully from {year}")
                break

    if artifacts is None:
        # Try default path
        print("\nTrying default model path...")
        artifacts = load_prediction_artifacts()

    # Test predictions
    results = test_predictions(df_261, df_101, artifacts)

    # Test material-level prediction
    test_material_level_prediction(df_261, df_101)

    # Save results
    if results is not None:
        output_path = os.path.join(DATA_DIR, "sample_test_results.csv")
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
