"""
Model Training Script for Material Yield Prediction System

SAP Manufacturing Logic:
- 261 = Goods Issue to Order = INPUT (raw materials CONSUMED from stock)
- 101 = Goods Receipt = OUTPUT (finished goods RECEIVED into stock)
- Yield = Output (101) / Input (261)
- Trains yield prediction models
- Saves artifacts for Streamlit app

Usage:
    python train_model.py
"""

import argparse
import os
import sys

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Import custom modules (use relative imports within src package)
from .data_preparation import prepare_full_dataset
from .model_training import train_yield_model, YieldPredictionModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Material Yield Prediction Model'
    )

    parser.add_argument(
        '--input261',
        type=str,
        default=os.path.join(DATA_DIR, '261.csv'),
        help='Path to 261 CSV file (INPUT - Goods Issue - raw materials consumed)'
    )

    parser.add_argument(
        '--output101',
        type=str,
        default=os.path.join(DATA_DIR, '101.csv'),
        help='Path to 101 CSV file (OUTPUT - Goods Receipt - finished goods produced)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=MODELS_DIR,
        help='Directory to save model artifacts'
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("MATERIAL YIELD PREDICTION - MODEL TRAINING")
    print("=" * 70)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Input (261 - Goods Issue): {args.input261}")
    print(f"  Output (101 - Goods Receipt): {args.output101}")
    print(f"  Output Dir: {args.output_dir}")

    # Validate files
    if not os.path.exists(args.input261):
        print(f"ERROR: File not found: {args.input261}")
        sys.exit(1)

    if not os.path.exists(args.output101):
        print(f"ERROR: File not found: {args.output101}")
        sys.exit(1)

    # Step 1: Data Preparation
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)

    try:
        df, encoders = prepare_full_dataset(
            input_261_path=args.input261,
            output_101_path=args.output101
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nData Summary:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    if 'Yield_Percentage' in df.columns:
        print(f"\nYield Statistics:")
        print(f"  Mean: {df['Yield_Percentage'].mean():.2f}%")
        print(f"  Std:  {df['Yield_Percentage'].std():.2f}%")
        print(f"  Min:  {df['Yield_Percentage'].min():.2f}%")
        print(f"  Max:  {df['Yield_Percentage'].max():.2f}%")

    # Step 2: Model Training
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)

    model = train_yield_model(df, encoders, save_path=args.output_dir)

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Model comparison
    print("\nModel Comparison:")
    print(model.get_model_comparison().to_string(index=False))

    # Feature importance
    importance = model.get_feature_importance()
    if len(importance) > 0:
        print("\nTop Features:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

    print(f"""
    Files Created:
    --------------
    - yield_model.joblib
    - all_models.joblib
    - encoders.joblib
    - feature_columns.joblib
    - metrics.joblib
    - scalers.joblib
    - test_results.csv
    - feature_importance.csv

    Next Step:
    ----------
    Run: streamlit run app.py
    """)


if __name__ == "__main__":
    main()
