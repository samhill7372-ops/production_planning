"""
Production Planning - Material Yield Prediction System

SAP Manufacturing Logic:
- 261 = Goods Issue to Order = INPUT (raw materials CONSUMED from stock)
- 101 = Goods Receipt = OUTPUT (finished goods RECEIVED into stock)
- Input and Output materials are DIFFERENT
- Join ONLY on MANUFACTURINGORDER
- Yield = Total_Output_BF (from 101) / Total_Input_BF (from 261)

Features:
- Multiple input materials support
- Output material simulation
- Yield prediction with model comparison
- Historical pattern analysis

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules from src/
from src.data_preparation import (
    prepare_full_dataset,
    prepare_full_dataset_with_raw,
    load_raw_csv_data,
    get_unique_materials_from_csv,
    get_historical_yield_by_material,
    get_available_years,
    load_precomputed_historical_data
)
from src.model_training import YieldPredictionModel, train_yield_model, OutputMaterialClassifier, train_output_classifier
from src.prediction_utils import (
    simulate_output_materials,
    simulate_output_materials_enhanced,
    get_prediction_summary,
    load_prediction_artifacts,
    get_encoder_categories,
    forward_predict,
    reverse_predict,
    estimate_material_waste,
    predict_output_material,
    get_top_recommendation,
    calculate_confidence_level,
    get_historical_kd_distribution,
    calculate_kd_output_with_wastage,
    get_material_level_forward_prediction,
    get_advanced_forward_prediction,
    get_material_level_forward_prediction_from_precomputed,
    get_advanced_forward_prediction_from_precomputed,
    get_historical_kd_distribution_from_precomputed,
    get_ks_grade_contributions_for_kd,
    get_ks_grade_contributions_for_kd_from_precomputed,
    get_batch_forward_prediction
)
from src.multi_output_prediction import run_prediction_all_orders as mo_predict_all_orders

# Page configuration
st.set_page_config(
    page_title="Production Planning - Yield Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .input-row {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

# Define paths based on folder structure
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def get_available_model_years():
    """Get list of years with trained models, including combined models."""
    models = []
    # Valid multi-year folder names
    multi_year_folders = {'2year', '5year'}

    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path):
                # Check if model exists in this folder
                if os.path.exists(os.path.join(item_path, "yield_model.joblib")):
                    if item.isdigit() or item in multi_year_folders:
                        models.append(item)

    # Sort: single years first (ascending), then multi-year
    def sort_key(x):
        if x.isdigit():
            return (0, int(x))
        return (1, int(''.join(filter(str.isdigit, x)) or '99'))

    return sorted(models, key=sort_key)


def get_model_display_name(model_id: str) -> str:
    """Convert model folder name to user-friendly display name."""
    display_names = {
        '2year': '2 years (2024 and 2025)',
        '5year': '5 years (historical)',
    }
    if model_id.isdigit():
        return model_id
    return display_names.get(model_id, model_id)


@st.cache_resource
def load_model_artifacts(model_year: str = None):
    """Load trained model and artifacts from models/ folder.

    Args:
        model_year: Specific year to load (e.g., '2024', '2025').
                    If None, loads from root models/ folder.
    """
    import json

    # Determine model path
    if model_year:
        base_path = os.path.join(MODELS_DIR, model_year)
    else:
        base_path = MODELS_DIR

    try:
        model_path = os.path.join(base_path, "yield_model.joblib")
        if not os.path.exists(model_path):
            return None

        model = joblib.load(model_path)
        encoders = joblib.load(os.path.join(base_path, "encoders.joblib")) if os.path.exists(os.path.join(base_path, "encoders.joblib")) else {}
        feature_columns = joblib.load(os.path.join(base_path, "feature_columns.joblib")) if os.path.exists(os.path.join(base_path, "feature_columns.joblib")) else []
        metrics = joblib.load(os.path.join(base_path, "metrics.joblib")) if os.path.exists(os.path.join(base_path, "metrics.joblib")) else {}
        test_results = pd.read_csv(os.path.join(base_path, "test_results.csv")) if os.path.exists(os.path.join(base_path, "test_results.csv")) else None

        # Load model metadata (includes training years)
        model_metadata = {}
        metadata_path = os.path.join(base_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        # Load Output Material Classifier (if available)
        classifier = None
        classifier_features = []
        classifier_metrics = {}
        classifier_source = "missing"

        classifier_path = os.path.join(base_path, "output_classifier.joblib")
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)
            classifier_source = "selected_model"
        else:
            # Fallback: use root-level classifier if year folder doesn't have one
            root_classifier_path = os.path.join(MODELS_DIR, "output_classifier.joblib")
            if os.path.exists(root_classifier_path):
                classifier = joblib.load(root_classifier_path)
                classifier_source = "root_fallback"

        clf_features_path = os.path.join(base_path, "classifier_features.joblib")
        if os.path.exists(clf_features_path):
            classifier_features = joblib.load(clf_features_path)
        else:
            # Fallback: use root-level classifier features
            root_clf_features_path = os.path.join(MODELS_DIR, "classifier_features.joblib")
            if os.path.exists(root_clf_features_path):
                classifier_features = joblib.load(root_clf_features_path)

        clf_metrics_path = os.path.join(base_path, "classifier_metrics.joblib")
        if os.path.exists(clf_metrics_path):
            classifier_metrics = joblib.load(clf_metrics_path)
        else:
            # Fallback: use root-level classifier metrics
            root_clf_metrics_path = os.path.join(MODELS_DIR, "classifier_metrics.joblib")
            if os.path.exists(root_clf_metrics_path):
                classifier_metrics = joblib.load(root_clf_metrics_path)

        return {
            'model': model,
            'encoders': encoders,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'test_results': test_results,
            'classifier': classifier,
            'classifier_features': classifier_features,
            'classifier_metrics': classifier_metrics,
            'classifier_source': classifier_source,
            'model_metadata': model_metadata
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_multi_output_model():
    """Load the multi-output yield prediction model and template."""
    model_dir = os.path.join(MODELS_DIR, "multi_output")
    model_path = os.path.join(model_dir, "baseline_yield_model.joblib")
    template_path = os.path.join(model_dir, "training_template.joblib")
    if not os.path.exists(model_path) or not os.path.exists(template_path):
        return None
    try:
        return {
            'model_bundle': joblib.load(model_path),
            'template_df': joblib.load(template_path),
        }
    except Exception as e:
        st.error(f"Error loading multi-output model: {e}")
        return None


@st.cache_resource
def load_kd_distribution_model():
    """Load the KD material distribution model and template."""
    model_dir = os.path.join(MODELS_DIR, "kd_distribution")
    bundle_path = os.path.join(model_dir, "kd_model_bundle.joblib")
    template_path = os.path.join(model_dir, "kd_training_template.joblib")
    history_path = os.path.join(model_dir, "kd_material_history.joblib")
    if not os.path.exists(bundle_path) or not os.path.exists(template_path):
        return None
    try:
        result = {
            'model_bundle': joblib.load(bundle_path),
            'template_df': joblib.load(template_path),
        }
        if os.path.exists(history_path):
            result['material_history'] = joblib.load(history_path)
        return result
    except Exception as e:
        st.error(f"Error loading KD distribution model: {e}")
        return None


@st.cache_data
def load_historical_data(selected_year: str = None):
    """Load and prepare historical data for the selected year.

    Args:
        selected_year: Year to load data for (e.g., '2024', '2025').
                      If None, loads all available years.

    Returns:
        Tuple of (historical_data, df_261_raw, df_101_raw, precomputed_data)
        - historical_data: Joined and encoded data (for distribution calculation)
        - df_261_raw: RAW 261 CSV data (for accurate BFIN totals) or None
        - df_101_raw: RAW 101 CSV data (for accurate BFOUT totals) or None
        - precomputed_data: Pre-computed statistics dict (for deployment) or None
    """
    # Determine which years to load
    if selected_year:
        if selected_year == '2year':
            years_to_load = ['2024', '2025']
        elif selected_year == '5year':
            years_to_load = get_available_years()
        else:
            years_to_load = [selected_year]
    else:
        years_to_load = get_available_years()

    # Load RAW CSV data for accurate totals (no cleaning/aggregation)
    df_261_raw, df_101_raw = None, None
    try:
        df_261_raw, df_101_raw = load_raw_csv_data(years=years_to_load)
    except Exception as e:
        print(f"Could not load raw CSV data: {e}")

    # Load pre-computed data (for deployment without CSV files)
    precomputed_data = None
    try:
        precomputed_data = load_precomputed_historical_data(year=selected_year)
    except Exception as e:
        print(f"Could not load pre-computed data: {e}")

    # Option 1: Load from year-specific CSV files
    try:
        df, encoders = prepare_full_dataset(years=years_to_load)
        print(f"Loaded historical data for {years_to_load}: {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        return df, df_261_raw, df_101_raw, precomputed_data
    except Exception as e:
        print(f"Could not load year-specific data: {e}")

    # Option 2: Try loading from generic CSV files (101.csv, 261.csv)
    try:
        csv_101 = os.path.join(DATA_DIR, "101.csv")
        csv_261 = os.path.join(DATA_DIR, "261.csv")
        if os.path.exists(csv_101) and os.path.exists(csv_261):
            df, encoders = prepare_full_dataset(csv_261, csv_101)
            return df, df_261_raw, df_101_raw, precomputed_data
    except Exception as e:
        print(f"Could not load from generic CSV: {e}")

    # Option 3: Load from pre-computed data (deployment mode)
    if precomputed_data is not None:
        summary_df = precomputed_data.get('summary_df')
        if summary_df is not None and len(summary_df) > 0:
            # Rename columns to match expected format
            if 'Mean_Yield' in summary_df.columns:
                summary_df = summary_df.rename(columns={
                    'Mean_Yield': 'Yield_Percentage',
                    'Order_Count': 'Historical_Orders'
                })
            print(f"Using pre-computed summary_df: {len(summary_df)} records")
            return summary_df, df_261_raw, df_101_raw, precomputed_data

    # Option 4: Load pre-computed historical summary from models/ folder
    try:
        hist_path = os.path.join(MODELS_DIR, "historical_summary.joblib")
        if os.path.exists(hist_path):
            historical_summary = joblib.load(hist_path)
            historical_summary = historical_summary.rename(columns={
                'Mean_Yield': 'Yield_Percentage',
                'Order_Count': 'Historical_Orders'
            })
            print(f"Loaded historical_summary.joblib: {len(historical_summary)} records")
            return historical_summary, df_261_raw, df_101_raw, precomputed_data
    except Exception as e:
        print(f"Could not load historical_summary: {e}")

    return None, df_261_raw, df_101_raw, precomputed_data


@st.cache_data
def load_dropdown_options():
    """Load options for dropdown menus from CSV files."""
    options = get_unique_materials_from_csv()
    # Remove KD (output) materials from Input_Material dropdown
    if options.get('Input_Material'):
        options['Input_Material'] = [m for m in options['Input_Material'] if 'KD' not in m.upper()]
    # Debug: print counts
    for key, values in options.items():
        if values:
            print(f"Loaded {len(values)} {key}")
    return options


# ============================================================================
# INPUT SECTION
# ============================================================================

def render_input_section():
    """Render the multi-input material section."""
    st.subheader("Input Materials (261 - Goods Issue - Raw Material Consumption)")

    # Initialize session state for input rows
    if 'input_materials' not in st.session_state:
        st.session_state.input_materials = [{}]

    # Load dropdown options
    options = load_dropdown_options()

    # Add/Remove buttons
    col_add, col_remove = st.columns([1, 1])
    with col_add:
        if st.button("+ Add Input Material", type="secondary"):
            st.session_state.input_materials.append({})
            st.rerun()
    with col_remove:
        if len(st.session_state.input_materials) > 1:
            if st.button("- Remove Last", type="secondary"):
                st.session_state.input_materials.pop()
                st.rerun()

    st.markdown("---")

    # Render input rows
    input_data_list = []

    for i, _ in enumerate(st.session_state.input_materials):
        with st.container():
            st.markdown(f"**Input Material {i + 1}**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Plant selection
                plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
                plant = st.selectbox(
                    "Plant",
                    options=plant_options if plant_options else ['1M02'],
                    key=f"plant_{i}"
                )

                # Material selection with search
                material_options = options.get('Input_Material', [])
                material_search = st.text_input(
                    "Search Material",
                    key=f"mat_search_{i}",
                    placeholder="Type to filter...",
                    help=f"Total materials available: {len(material_options)}"
                )

                if material_search:
                    filtered = [m for m in material_options if material_search.upper() in str(m).upper()]
                else:
                    filtered = material_options  # Show ALL materials

                material = st.selectbox(
                    f"Material ({len(filtered)} available)",
                    options=filtered if filtered else ['No materials found'],
                    key=f"material_{i}"
                )

            with col2:
                # Thickness
                thickness_options = options.get('Input_Thickness', [4, 6, 8])
                thickness = st.selectbox(
                    "Thickness",
                    options=thickness_options if thickness_options else [4, 6, 8],
                    key=f"thickness_{i}"
                )

                # Specie
                specie_options = options.get('Input_Specie', ['SM', 'AS', 'WO'])
                specie = st.selectbox(
                    "Specie",
                    options=specie_options if specie_options else ['SM'],
                    key=f"specie_{i}"
                )

            with col3:
                # Grade
                grade_options = options.get('Input_Grade', ['2C', '1C', '3A'])
                grade = st.selectbox(
                    "Grade",
                    options=grade_options if grade_options else ['2C'],
                    key=f"grade_{i}"
                )

                # Movement Type (for reference)
                st.selectbox(
                    "Movement Type",
                    options=['261 (Goods Issue - Input)'],
                    index=0,
                    disabled=True,
                    key=f"mvt_{i}",
                    help="261 = Raw materials consumed from stock (INPUT)"
                )

            with col4:
                # Quantity (BFIN)
                quantity = st.number_input(
                    "Quantity (BFIN)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0,
                    key=f"quantity_{i}"
                )

                # Length/Width (optional)
                length = st.number_input(
                    "Avg Length",
                    min_value=0.0,
                    value=96.0,
                    key=f"length_{i}"
                )

            # Store input data
            input_data_list.append({
                'Input_Plant': plant,
                'Input_Material': material,
                'Input_Thickness': thickness,
                'Input_Specie': specie,
                'Input_Grade': grade,
                'Input_Length': length,
                'Input_Width': 8.0,  # Default
                'Total_Input_BF': quantity
            })

        st.markdown("---")

    return input_data_list


# ============================================================================
# PREDICTION RESULTS
# ============================================================================

def render_prediction_summary(summary: Dict, output_results: pd.DataFrame):
    """Render the prediction summary section."""
    st.subheader("Prediction Summary")

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Input Quantity",
            value=f"{summary['total_input_bf']:,.0f} BF"
        )

    with col2:
        st.metric(
            label="Possible Output Materials",
            value=f"{summary['num_outputs']}"
        )

    with col3:
        if summary['best_output']:
            st.metric(
                label="Best Output Material",
                value=summary['best_output'][:15] + "..." if len(str(summary['best_output'])) > 15 else summary['best_output']
            )

    with col4:
        st.metric(
            label="Best Yield",
            value=f"{summary['best_yield_pct']:.1f}%",
            delta=f"{summary['best_output_bf']:,.0f} BF output"
        )

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="Average Yield",
            value=f"{summary['avg_yield_pct']:.1f}%"
        )

    with col6:
        st.metric(
            label="Models Evaluated",
            value=f"{summary['models_evaluated']}"
        )

    with col7:
        st.metric(
            label="Best Model R²",
            value=f"{summary['best_r2']:.4f}"
        )

    with col8:
        st.metric(
            label="Lowest RMSE",
            value=f"{summary['lowest_rmse']:.2f}%"
        )

    # Multi-input advantage indicator
    if summary.get('multi_input_advantage'):
        st.success("Multi-Input Model Advantage: More output options identified!")


def render_reverse_prediction_section(model, encoders, feature_columns, options):
    """Render the reverse prediction section - calculate required input from desired output."""
    st.subheader("Reverse Prediction: Calculate Required Input")
    st.caption("Specify your desired output quantity and we'll calculate how much raw material you need")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Desired Output**")
        desired_output_bf = st.number_input(
            "Target Output (BF)",
            min_value=0.0,
            max_value=10000000.0,
            value=5000.0,
            step=500.0,
            help="How much finished product do you need?"
        )

        safety_margin = st.slider(
            "Safety Margin (%)",
            min_value=0.0,
            max_value=25.0,
            value=5.0,
            step=1.0,
            help="Extra buffer to account for yield variability"
        )

    with col2:
        st.markdown("**Input Material Characteristics**")

        # Plant selection
        plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
        rev_plant = st.selectbox(
            "Plant",
            options=plant_options if plant_options else ['1M02'],
            key="rev_plant"
        )

        # Material selection
        material_options = options.get('Input_Material', [])
        rev_mat_search = st.text_input(
            "Search Material",
            key="rev_mat_search",
            placeholder="Type to filter..."
        )

        if rev_mat_search:
            filtered = [m for m in material_options if rev_mat_search.upper() in str(m).upper()]
        else:
            filtered = material_options[:100]  # Limit initial display

        rev_material = st.selectbox(
            f"Input Material ({len(filtered)} shown)",
            options=filtered if filtered else ['No materials found'],
            key="rev_material"
        )

    # Additional material properties
    col3, col4, col5 = st.columns(3)

    with col3:
        thickness_options = options.get('Input_Thickness', [4, 6, 8])
        rev_thickness = st.selectbox(
            "Thickness",
            options=thickness_options if thickness_options else [4, 6, 8],
            key="rev_thickness"
        )

    with col4:
        specie_options = options.get('Input_Specie', ['SM', 'AS', 'WO'])
        rev_specie = st.selectbox(
            "Specie",
            options=specie_options if specie_options else ['SM'],
            key="rev_specie"
        )

    with col5:
        grade_options = options.get('Input_Grade', ['2C', '1C', '3A'])
        rev_grade = st.selectbox(
            "Grade",
            options=grade_options if grade_options else ['2C'],
            key="rev_grade"
        )

    # Calculate button
    if st.button("Calculate Required Input", type="primary", key="reverse_calc"):
        if desired_output_bf <= 0:
            st.error("Please enter a target output quantity greater than 0")
        else:
            # Build input data dict
            input_data = {
                'Input_Plant': rev_plant,
                'Input_Material': rev_material,
                'Input_Thickness': rev_thickness,
                'Input_Specie': rev_specie,
                'Input_Grade': rev_grade,
                'Input_Length': 96.0,
                'Input_Width': 8.0,
                'Total_Input_BF': 10000  # Placeholder for yield calculation
            }

            # Run reverse prediction
            result = reverse_predict(
                model=model,
                required_output_bf=desired_output_bf,
                input_data=input_data,
                encoders=encoders,
                feature_columns=feature_columns,
                safety_margin_pct=safety_margin
            )

            # Display results
            st.markdown("---")
            st.markdown("### Results")

            # Main metrics
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.metric(
                    label="Required Input (with safety)",
                    value=f"{result['required_input_bf']:,.0f} BF",
                    help=f"Includes {safety_margin}% safety margin"
                )

            with col_r2:
                st.metric(
                    label="Required Input (minimum)",
                    value=f"{result['required_input_no_safety_bf']:,.0f} BF",
                    help="Without safety margin"
                )

            with col_r3:
                st.metric(
                    label="Expected Yield",
                    value=f"{result['predicted_yield_pct']:.1f}%"
                )

            # Visual explanation
            waste_bf = result['required_input_bf'] - desired_output_bf
            waste_pct = 100 - result['predicted_yield_pct']

            st.info(f"""
            **Planning Summary:**

            To produce **{desired_output_bf:,.0f} BF** of finished product:

            - You need approximately **{result['required_input_bf']:,.0f} BF** of raw material ({rev_material})
            - Expected yield is **{result['predicted_yield_pct']:.1f}%**
            - Approximately **{waste_bf:,.0f} BF** ({waste_pct:.1f}%) will be lost during production
            - Safety margin of **{safety_margin}%** is included to account for variability
            """)

            # Breakdown table
            with st.expander("Detailed Breakdown"):
                breakdown_data = {
                    'Metric': [
                        'Target Output',
                        'Predicted Yield',
                        'Minimum Required Input',
                        'Safety Margin',
                        'Recommended Input (with safety)',
                        'Expected Waste'
                    ],
                    'Value': [
                        f"{desired_output_bf:,.0f} BF",
                        f"{result['predicted_yield_pct']:.1f}%",
                        f"{result['required_input_no_safety_bf']:,.0f} BF",
                        f"{safety_margin}%",
                        f"{result['required_input_bf']:,.0f} BF",
                        f"{waste_bf:,.0f} BF ({waste_pct:.1f}%)"
                    ]
                }
                st.table(pd.DataFrame(breakdown_data))


def render_sample_data_test_section(model, encoders, feature_columns):
    """Test model predictions using sample data files (sample_261.csv and sample_101.csv).

    This section allows testing the model against controlled sample data to:
    - Verify yield predictions accuracy
    - Compare actual vs predicted output materials
    - Identify prediction errors and issues
    """
    st.subheader("Sample Data Test")
    st.caption("Test model predictions against sample_261.csv and sample_101.csv")

    # Load sample data files
    sample_261_path = os.path.join(DATA_DIR, "sample_261.csv")
    sample_101_path = os.path.join(DATA_DIR, "sample_101.csv")

    # Check if files exist
    if not os.path.exists(sample_261_path):
        st.error(f"sample_261.csv not found at {sample_261_path}")
        st.info("Create sample data files first using scripts/test_sample_data.py or manually.")
        return

    if not os.path.exists(sample_101_path):
        st.error(f"sample_101.csv not found at {sample_101_path}")
        return

    # Load sample data
    df_261 = pd.read_csv(sample_261_path)
    df_101 = pd.read_csv(sample_101_path)

    # Data Summary
    st.markdown("### Sample Data Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Orders", df_261['MANUFACTURINGORDER'].nunique())
    with col2:
        st.metric("Input Materials", df_261['MATERIAL'].nunique())
    with col3:
        st.metric("Total Input BF", f"{df_261['BFIN'].sum():,.0f}")
    with col4:
        st.metric("Total Output BF", f"{df_101['BFOUT'].sum():,.0f}")

    # Calculate actual yields
    input_by_order = df_261.groupby('MANUFACTURINGORDER')['BFIN'].sum()
    output_by_order = df_101.groupby('MANUFACTURINGORDER')['BFOUT'].sum()

    actual_yields = pd.DataFrame({
        'Input_BF': input_by_order,
        'Output_BF': output_by_order
    }).fillna(0)
    actual_yields['Actual_Yield_Pct'] = (actual_yields['Output_BF'] / actual_yields['Input_BF'] * 100).round(2)

    overall_actual_yield = actual_yields['Output_BF'].sum() / actual_yields['Input_BF'].sum() * 100

    st.markdown("---")

    # Run predictions button
    if st.button("Run Predictions", type="primary", use_container_width=True):
        results = []
        orders = df_261['MANUFACTURINGORDER'].unique()

        progress_bar = st.progress(0)

        for i, order in enumerate(orders):
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
                predicted_yield = 0
                predicted_output_bf = 0

            # Actual output materials
            actual_materials = order_101['MATERIAL'].unique().tolist() if len(order_101) > 0 else []

            yield_error = predicted_yield - actual_yield

            results.append({
                'Order': order,
                'Input Material': input_material,
                'Species': input_specie,
                'Plant': input_plant,
                'Input BF': total_input_bf,
                'Actual Output BF': total_output_bf,
                'Predicted Output BF': round(predicted_output_bf, 0),
                'Actual Yield %': round(actual_yield, 2),
                'Predicted Yield %': round(predicted_yield, 2),
                'Yield Error %': round(yield_error, 2),
                'Output Materials': len(actual_materials)
            })

            progress_bar.progress((i + 1) / len(orders))

        progress_bar.empty()

        # Display results
        st.markdown("### Prediction Results")

        df_results = pd.DataFrame(results)

        # Summary metrics
        mae = df_results['Yield Error %'].abs().mean()
        rmse = np.sqrt((df_results['Yield Error %'] ** 2).mean())
        avg_predicted = df_results['Predicted Yield %'].mean()
        avg_actual = df_results['Actual Yield %'].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Abs Error", f"{mae:.2f}%")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}%")
        with col3:
            st.metric("Avg Predicted Yield", f"{avg_predicted:.2f}%")
        with col4:
            st.metric("Avg Actual Yield", f"{avg_actual:.2f}%")

        st.markdown("---")

        # Results table with color coding
        st.markdown("### Detailed Results by Order")

        # Style the dataframe
        def color_error(val):
            if abs(val) < 3:
                return 'background-color: #d4edda'  # Green
            elif abs(val) < 5:
                return 'background-color: #fff3cd'  # Yellow
            else:
                return 'background-color: #f8d7da'  # Red

        styled_df = df_results.style.applymap(color_error, subset=['Yield Error %'])
        st.dataframe(styled_df, use_container_width=True)

        # Per-order details in expandable sections
        st.markdown("### Order Details")

        for r in results:
            order_261 = df_261[df_261['MANUFACTURINGORDER'] == r['Order']]
            order_101 = df_101[df_101['MANUFACTURINGORDER'] == r['Order']]

            error_color = "🟢" if abs(r['Yield Error %']) < 3 else ("🟡" if abs(r['Yield Error %']) < 5 else "🔴")

            with st.expander(f"{error_color} {r['Order']} - {r['Input Material']} ({r['Species']}) - Error: {r['Yield Error %']:+.2f}%"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Input (261)**")
                    st.write(f"- Material: {r['Input Material']}")
                    st.write(f"- Species: {r['Species']}")
                    st.write(f"- Plant: {r['Plant']}")
                    st.write(f"- Input BF: {r['Input BF']:,.0f}")
                    st.write(f"- Grades: {', '.join(order_261['TALLYGRADE'].unique())}")

                with col2:
                    st.markdown("**Output (101)**")
                    st.write(f"- Actual Output BF: {r['Actual Output BF']:,.0f}")
                    st.write(f"- Predicted Output BF: {r['Predicted Output BF']:,.0f}")
                    st.write(f"- Actual Yield: {r['Actual Yield %']:.2f}%")
                    st.write(f"- Predicted Yield: {r['Predicted Yield %']:.2f}%")
                    st.write(f"- **Error: {r['Yield Error %']:+.2f}%**")

                # Output materials
                st.markdown("**Output Materials:**")
                output_mat_summary = order_101.groupby('MATERIAL')['BFOUT'].sum().sort_values(ascending=False)
                st.dataframe(output_mat_summary.reset_index().rename(columns={'MATERIAL': 'Output Material', 'BFOUT': 'BF Output'}), use_container_width=True)

        # Save results option
        if st.button("Save Results to CSV"):
            output_path = os.path.join(DATA_DIR, "sample_test_results.csv")
            df_results.to_csv(output_path, index=False)
            st.success(f"Results saved to {output_path}")


def render_material_level_forward_prediction_section(options, df_261_raw, df_101_raw, precomputed_data=None):
    """Render the Material Level Forward Prediction (Yield Recommendation Engine).

    This mode allows users to:
    - Enter KS material, plant, and input BF quantity
    - Get predicted total output BF based on historical yield
    - Get distribution across KD materials (summing to 100%)

    Uses pre-computed data when raw CSV files are not available (deployment mode).
    """
    st.subheader("Material Level Forward Prediction")
    st.caption("Enter KS material, plant, and quantity to predict KD output distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**KS Material (Input)**")

        # Plant selection
        plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
        selected_plant = st.selectbox(
            "Plant *",
            options=plant_options,
            key="mlfp_plant"
        )

        # Material selection with search
        material_options = options.get('Input_Material', [])
        mat_search = st.text_input(
            "Search KS Material",
            key="mlfp_mat_search",
            placeholder="Type to filter..."
        )

        if mat_search:
            filtered = [m for m in material_options if mat_search.upper() in str(m).upper()]
        else:
            filtered = material_options

        selected_material = st.selectbox(
            f"KS Material * ({len(filtered)} available)",
            options=filtered if filtered else ['No materials found'],
            key="mlfp_material"
        )

    with col2:
        st.markdown("**Input Quantity**")
        input_bf = st.number_input(
            "Input BF *",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            key="mlfp_input_bf",
            help="Board feet of KS material to process"
        )

        min_orders = st.number_input(
            "Minimum Order Threshold",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="mlfp_min_orders",
            help="Only show KD materials with more than this many historical orders"
        )

    st.markdown("---")

    if st.button("Predict KD Output", type="primary", key="mlfp_predict_btn", use_container_width=True):
        # Check if we have data available (either raw CSV or pre-computed)
        has_raw_data = df_261_raw is not None and df_101_raw is not None
        has_precomputed = precomputed_data is not None and 'material_stats' in precomputed_data

        if not has_raw_data and not has_precomputed:
            st.error("No data available. Please ensure data files are loaded or pre-computed data exists.")
            return

        if selected_material == 'No materials found':
            st.error("Please select a valid KS material.")
            return

        with st.spinner("Calculating prediction..."):
            # Use pre-computed data when raw CSV is not available
            if has_raw_data:
                result = get_material_level_forward_prediction(
                    ks_material=selected_material,
                    plant=selected_plant,
                    input_bf=input_bf,
                    df_261_raw=df_261_raw,
                    df_101_raw=df_101_raw,
                    min_order_count=min_orders
                )
            else:
                result = get_material_level_forward_prediction_from_precomputed(
                    ks_material=selected_material,
                    plant=selected_plant,
                    input_bf=input_bf,
                    material_stats=precomputed_data['material_stats'],
                    min_order_count=min_orders
                )
            st.session_state.mlfp_result = result
            st.session_state.mlfp_run = True

    # Display Results
    if st.session_state.get('mlfp_run'):
        result = st.session_state.get('mlfp_result', {})

        if 'error' in result:
            st.warning(result['error'])
            return

        st.markdown("---")
        st.header("Prediction Results")

        # Summary metrics row 1 - Input/Output
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Input BF", f"{result['input_bf']:,.0f}")
        with col_s2:
            st.metric("Predicted Output BF", f"{result['predicted_output_bf']:,.0f}")
        with col_s3:
            st.metric("Yield %", f"{result['historical_yield_pct']:.1f}%")

        # Summary metrics row 2 - Historical context
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.metric("Historical Input BF", f"{result['total_hist_input_bf']:,.0f}")
        with col_h2:
            st.metric("Historical Output BF", f"{result['total_hist_output_bf']:,.0f}")
        with col_h3:
            st.metric("Historical Orders", f"{result['total_orders']:,}")

        st.markdown("---")
        st.subheader(f"KD Material Distribution ({result['kd_materials_count']} materials)")

        distribution = result.get('kd_distribution', [])
        if distribution:
            df_dist = pd.DataFrame(distribution)

            # Format display table
            df_display = pd.DataFrame({
                'KD Material': df_dist['KD_Material'],
                'Contribution %': df_dist['Contribution_Pct'].apply(lambda x: f"{x:.1f}%"),
                'Expected BF Output': df_dist['Expected_BF_Output'].apply(lambda x: f"{x:,.0f}"),
                'Historical Orders': df_dist['Order_Count'],
                'Historical BF': df_dist['Historical_BF_Output'].apply(lambda x: f"{x:,.0f}")
            })

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Totals row
            total_pct = sum(d['Contribution_Pct'] for d in distribution)
            total_bf = sum(d['Expected_BF_Output'] for d in distribution)
            st.markdown(f"**Total: {total_pct:.1f}% | {total_bf:,.0f} BF**")

            # Download button
            export_df = pd.DataFrame(distribution)
            export_df['KS_Material'] = result['ks_material']
            export_df['Plant'] = result['plant']
            export_df['Input_BF'] = result['input_bf']
            export_df['Predicted_Total_Output_BF'] = result['predicted_output_bf']
            export_df['Yield_Pct'] = result['historical_yield_pct']

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download KD Distribution (CSV)",
                data=csv,
                file_name=f"kd_prediction_{result['ks_material']}_{result['plant']}.csv",
                mime="text/csv"
            )


def render_batch_prediction_section(options, df_261_raw, df_101_raw, model, encoders, feature_columns, precomputed_data=None):
    """Render the Batch Prediction section for multiple materials.

    Predict yield for multiple materials at once with full KD breakdown.
    Uses historical averages for each material (same as single prediction).
    """
    st.subheader("Batch Prediction")
    st.caption("Predict yield for multiple materials at once with full KD breakdown")

    st.markdown("**Add multiple materials for batch yield prediction**")
    st.info("Uses historical averages for each material (same as single Advanced Forward Prediction)")

    # Initialize session state for batch materials
    if 'batch_forward_materials' not in st.session_state:
        st.session_state.batch_forward_materials = [0]  # Track number of rows

    # Add/Remove buttons
    col_add, col_remove, col_clear = st.columns(3)
    with col_add:
        if st.button("+ Add Material Row", key="batch_add"):
            st.session_state.batch_forward_materials.append(len(st.session_state.batch_forward_materials))
            st.rerun()
    with col_remove:
        if st.button("- Remove Last Row", key="batch_remove"):
            if len(st.session_state.batch_forward_materials) > 1:
                st.session_state.batch_forward_materials.pop()
                st.rerun()
    with col_clear:
        if st.button("Clear All", key="batch_clear"):
            st.session_state.batch_forward_materials = [0]
            if 'batch_results' in st.session_state:
                del st.session_state.batch_results
            st.rerun()

    # Get option lists
    plant_options = options.get('Input_Plant', ['1Y01'])
    material_options = options.get('Input_Material', [])

    # Render simplified input rows (only Plant, Material, Input BF)
    st.markdown("---")
    for i in range(len(st.session_state.batch_forward_materials)):
        c1, c2, c3 = st.columns([2, 3, 2])
        with c1:
            st.selectbox(f"Plant {i+1}", plant_options, key=f"batch_plant_{i}")
        with c2:
            st.selectbox(f"Material {i+1}", material_options, key=f"batch_material_{i}")
        with c3:
            st.number_input(f"Input BF {i+1}", min_value=0.0, value=10000.0, step=1000.0, key=f"batch_bf_{i}")

    st.markdown("---")

    # Batch Predict button
    if st.button("Predict All Materials", type="primary", key="batch_predict_btn", use_container_width=True):
        if model is None:
            st.error("ML model not available.")
        elif df_261_raw is None or df_101_raw is None:
            st.error("Historical data not available for KD breakdown.")
        else:
            # Collect all material inputs (simplified - only plant, material, input_bf)
            materials_list = []
            for i in range(len(st.session_state.batch_forward_materials)):
                materials_list.append({
                    'plant': st.session_state.get(f"batch_plant_{i}", '1Y01'),
                    'material': st.session_state.get(f"batch_material_{i}", ''),
                    'input_bf': st.session_state.get(f"batch_bf_{i}", 10000.0)
                })

            with st.spinner("Running batch predictions with KD breakdown..."):
                results = get_batch_forward_prediction(
                    materials_list,
                    df_261_raw,
                    df_101_raw,
                    model,
                    encoders,
                    feature_columns,
                    min_order_count=5
                )
                st.session_state.batch_results = results

    # Display batch results
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        st.markdown("### Batch Prediction Results")

        batch_results = st.session_state.batch_results

        # Overall summary
        total_input_bf = sum(r.get('input_bf', 0) for r in batch_results)
        total_output_bf = sum(r.get('predicted_output_bf', 0) for r in batch_results)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Materials", len(batch_results))
        with col_s2:
            st.metric("Total Input BF", f"{total_input_bf:,.0f}")
        with col_s3:
            st.metric("Total Predicted Output BF", f"{total_output_bf:,.0f}")

        # Display each material with expandable KD breakdown
        for i, result in enumerate(batch_results):
            material_name = result.get('input_material', result.get('ks_material', f'Material {i+1}'))
            plant_name = result.get('input_plant', result.get('plant', ''))

            if 'error' in result:
                st.warning(f"**{material_name} @ {plant_name}**: {result['error']}")
                continue

            with st.expander(f"**{material_name}** @ {plant_name} - Yield: {result.get('ml_yield_pct', 0):.1f}%", expanded=False):
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ML Predicted Yield", f"{result.get('ml_yield_pct', 0):.1f}%")
                with col2:
                    st.metric("Input BF", f"{result.get('input_bf', 0):,.0f}")
                with col3:
                    st.metric("Predicted Output BF", f"{result.get('predicted_output_bf', 0):,.0f}")

                # Historical context
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("Historical Yield", f"{result.get('historical_yield_pct', 0):.1f}%")
                with col5:
                    st.metric("Historical Orders", f"{result.get('historical_orders', 0):,}")
                with col6:
                    st.metric("Historical Output BF", f"{result.get('total_hist_output_bf', 0):,.0f}")

                # KD Distribution table
                kd_dist = result.get('kd_distribution', [])
                if kd_dist:
                    st.markdown("**KD Material Distribution:**")
                    kd_df = pd.DataFrame(kd_dist)
                    # Select relevant columns
                    display_cols = ['KD_Material', 'Contribution_Pct', 'Expected_BF_Output', 'Historical_Orders', 'Historical_BF_Output']
                    display_cols = [c for c in display_cols if c in kd_df.columns]
                    if display_cols:
                        st.dataframe(kd_df[display_cols], use_container_width=True)

        # CSV export for summary
        st.markdown("---")
        summary_data = []
        for result in batch_results:
            if 'error' not in result:
                summary_data.append({
                    'Plant': result.get('input_plant', result.get('plant', '')),
                    'Material': result.get('input_material', result.get('ks_material', '')),
                    'Input_BF': result.get('input_bf', 0),
                    'ML_Yield_Pct': result.get('ml_yield_pct', 0),
                    'Predicted_Output_BF': result.get('predicted_output_bf', 0),
                    'Historical_Yield_Pct': result.get('historical_yield_pct', 0),
                    'Historical_Orders': result.get('historical_orders', 0)
                })
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                "Download Summary CSV",
                csv,
                "batch_predictions_summary.csv",
                "text/csv",
                key="batch_download"
            )


def render_multi_output_prediction_section():
    """Render the Multi-Output Prediction section.

    Users upload a 261 consumption CSV, select a manufacturing order,
    and get detailed output board distribution (Grade/Length/Width/Boards).
    """
    st.subheader("Multi-Output Prediction")
    st.caption("Upload 261 consumption data to predict detailed output board distribution")

    # Load multi-output model
    mo_artifacts = load_multi_output_model()
    if mo_artifacts is None:
        st.error("Multi-output model not found. Ensure baseline_yield_model.joblib and training_template.joblib exist in models/multi_output/")
        return

    model_bundle = mo_artifacts['model_bundle']
    template_df = mo_artifacts['template_df']

    st.info("""
    **Required CSV columns:** MANUFACTURINGORDER, BFIN, TALLYLENGTH, TALLYWIDTH
    **Recommended columns:** MATERIALSPECIE, MATERIALTHICKNESS, PLANT, TALLYGRADE, GOODSMOVEMENTTYPE
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your 261 consumption CSV file:",
        type=["csv"],
        key="mo_file_upload",
        help="CSV should contain 261 consumption records with columns: MANUFACTURINGORDER, BFIN, TALLYLENGTH, TALLYWIDTH, etc."
    )

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ["BFIN", "TALLYLENGTH", "TALLYWIDTH", "MANUFACTURINGORDER"]
        recommended_cols = ["MATERIALSPECIE", "MATERIALTHICKNESS", "PLANT", "TALLYGRADE", "GOODSMOVEMENTTYPE"]

        missing_required = [col for col in required_cols if col not in df_upload.columns]
        missing_recommended = [col for col in recommended_cols if col not in df_upload.columns]

        if missing_required:
            st.error(f"Missing required columns: {', '.join(missing_required)}")
            return

        if missing_recommended:
            st.warning(f"Missing recommended columns (predictions may be less accurate): {', '.join(missing_recommended)}")

        # Extract unique manufacture orders and compute stats
        unique_orders = sorted(df_upload["MANUFACTURINGORDER"].unique())
        order_stats = df_upload.groupby("MANUFACTURINGORDER").agg(
            rows=("MANUFACTURINGORDER", "count"),
            bfin=("BFIN", "sum")
        )

        # Manufacture order dropdown
        st.markdown("---")
        st.markdown("**Select Manufacturing Order**")
        selected_order = st.selectbox(
            "Choose a manufacturing order to analyze:",
            options=unique_orders,
            format_func=lambda x: f"{x} ({order_stats.loc[x, 'rows']:,.0f} rows, {order_stats.loc[x, 'bfin']:,.0f} BFIN)",
            key="mo_order_select"
        )

        # Filter dataframe to selected order
        df_filtered = df_upload[df_upload["MANUFACTURINGORDER"] == selected_order]

        st.markdown("**Input Data Preview**")
        st.dataframe(df_filtered.head(10), use_container_width=True)

        total_bfin_preview = df_filtered["BFIN"].sum()
        st.caption(f"Selected order: {selected_order} | Rows: {len(df_filtered):,} | Total BFIN: {total_bfin_preview:,.0f}")

        # Run prediction
        st.markdown("---")
        if st.button("Run Multi-Output Prediction", type="primary", key="mo_predict_btn", use_container_width=True):
            progress_bar = st.progress(0, text="Starting prediction...")
            result_df, summary = mo_predict_all_orders(
                df_filtered,
                model_bundle,
                template_df,
                progress_bar
            )
            progress_bar.empty()

            if result_df is not None and summary is not None:
                st.session_state.mo_result = result_df
                st.session_state.mo_summary = summary

        # Display results (persistent via session state)
        if 'mo_result' in st.session_state and st.session_state.mo_result is not None:
            result_df = st.session_state.mo_result
            summary = st.session_state.mo_summary

            st.markdown("---")

            # Summary metrics
            st.markdown("### Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Orders Processed", f"{summary['num_orders']:,}")
            col2.metric("Total BFIN", f"{summary['total_bfin']:,.0f}")
            col3.metric("Avg Yield Factor", f"{summary['yield_factor']:.2%}")
            col4.metric("Predicted Output", f"{summary['predicted_output']:,.0f}")
            col5.metric("Total Boards", f"{summary['total_boards']:,}")

            st.markdown("---")

            # Results table and chart side by side
            col_table, col_chart = st.columns([1, 1])

            with col_table:
                st.markdown("**Predicted Distribution**")
                st.dataframe(result_df, use_container_width=True, height=400)

            with col_chart:
                st.markdown("**Top Bins by Volume**")
                if not result_df.empty:
                    chart_df = result_df.head(15).copy()
                    chart_df["Bin"] = chart_df["Grade"] + "_" + chart_df["Length"].astype(str) + "_" + chart_df["Width"].astype(str)
                    st.bar_chart(chart_df.set_index("Bin")["Boards"])

            # Download button
            st.markdown("---")
            csv_output = result_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_output,
                file_name="predicted_output.csv",
                mime="text/csv",
                key="mo_download"
            )


def render_advanced_forward_prediction_section(options, df_261_raw, df_101_raw, model, encoders, feature_columns, precomputed_data=None):
    """Render the Advanced Forward Prediction (ML + Statistical Hybrid).

    Uses ML model for yield prediction + statistical distribution for KD materials.
    Uses pre-computed data when raw CSV files are not available (deployment mode).
    """
    st.subheader("Advanced Forward Prediction")
    st.caption("ML model for yield prediction + statistical distribution for KD materials")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**KS Material (Input)**")

        # Plant selection
        plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
        selected_plant = st.selectbox(
            "Plant *",
            options=plant_options,
            key="afp_plant"
        )

        # Material selection with search
        material_options = options.get('Input_Material', [])
        mat_search = st.text_input(
            "Search KS Material",
            key="afp_mat_search",
            placeholder="Type to filter..."
        )

        if mat_search:
            filtered = [m for m in material_options if mat_search.upper() in str(m).upper()]
        else:
            filtered = material_options

        selected_material = st.selectbox(
            f"KS Material * ({len(filtered)} available)",
            options=filtered if filtered else ['No materials found'],
            key="afp_material"
        )

    with col2:
        st.markdown("**Input Quantity**")
        input_bf = st.number_input(
            "Input BF *",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            key="afp_input_bf",
            help="Board feet of KS material to process"
        )

        min_orders = st.number_input(
            "Minimum Order Threshold",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="afp_min_orders",
            help="Only show KD materials with more than this many historical orders"
        )

    st.markdown("---")

    if st.button("Predict with ML Model", type="primary", key="afp_predict_btn", use_container_width=True):
        # Check if we have data available (either raw CSV or pre-computed)
        has_raw_data = df_261_raw is not None and df_101_raw is not None
        has_precomputed = precomputed_data is not None and 'material_stats' in precomputed_data

        if not has_raw_data and not has_precomputed:
            st.error("No data available. Please ensure data files are loaded or pre-computed data exists.")
            return

        if selected_material == 'No materials found':
            st.error("Please select a valid KS material.")
            return

        if model is None:
            st.error("ML model not available. Please ensure the model is trained.")
            return

        with st.spinner("Running ML prediction..."):
            # Use pre-computed data when raw CSV is not available
            if has_raw_data:
                result = get_advanced_forward_prediction(
                    ks_material=selected_material,
                    plant=selected_plant,
                    input_bf=input_bf,
                    df_261_raw=df_261_raw,
                    df_101_raw=df_101_raw,
                    model=model,
                    encoders=encoders,
                    feature_columns=feature_columns,
                    min_order_count=min_orders
                )
            else:
                result = get_advanced_forward_prediction_from_precomputed(
                    ks_material=selected_material,
                    plant=selected_plant,
                    input_bf=input_bf,
                    material_stats=precomputed_data['material_stats'],
                    model=model,
                    encoders=encoders,
                    feature_columns=feature_columns,
                    min_order_count=min_orders
                )
            st.session_state.afp_result = result
            st.session_state.afp_run = True

    # Display Results
    if st.session_state.get('afp_run'):
        result = st.session_state.get('afp_result', {})

        if 'error' in result:
            st.warning(result['error'])
            return

        st.markdown("---")
        st.header("Prediction Results")

        # Show prediction method
        st.info(f"**Prediction Method:** {result.get('prediction_method', 'ML + Statistical')}")

        # Summary metrics row 1 - Yield comparison
        col_y1, col_y2, col_y3 = st.columns(3)
        with col_y1:
            st.metric("ML Predicted Yield", f"{result['ml_yield_pct']:.1f}%")
        with col_y2:
            st.metric("Historical Yield", f"{result['historical_yield_pct']:.1f}%")
        with col_y3:
            diff = result['ml_yield_pct'] - result['historical_yield_pct']
            st.metric("Difference", f"{diff:+.1f}%")

        # Summary metrics row 2 - Input/Output
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Input BF", f"{result['input_bf']:,.0f}")
        with col_s2:
            st.metric("Predicted Output BF", f"{result['predicted_output_bf']:,.0f}")
        with col_s3:
            st.metric("ML Confidence", result.get('ml_confidence', 'N/A'))

        # Summary metrics row 3 - Historical context
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.metric("Historical Input BF", f"{result['total_hist_input_bf']:,.0f}")
        with col_h2:
            st.metric("Historical Output BF", f"{result['total_hist_output_bf']:,.0f}")
        with col_h3:
            st.metric("Historical Orders", f"{result['total_orders']:,}")

        st.markdown("---")
        st.subheader(f"KD Material Distribution ({result['kd_materials_count']} materials)")
        st.caption("Click checkbox to expand grade breakdown (PR, 1C, 2C, etc.) with dimensions")

        distribution = result.get('kd_distribution', [])
        if distribution:
            # Check if we can show drill-down (need raw data or precomputed grade data)
            has_raw_data = df_261_raw is not None and df_101_raw is not None
            has_precomputed_grade = precomputed_data is not None and 'grade_breakdown_data' in precomputed_data
            has_grade_data = has_raw_data or has_precomputed_grade

            # Tree table header
            header_cols = st.columns([0.03, 0.15, 0.10, 0.12, 0.10, 0.12, 0.12, 0.12, 0.14])
            header_cols[0].markdown("")  # Expand toggle
            header_cols[1].markdown("**KD Material**")
            header_cols[2].markdown("**Contrib %**")
            header_cols[3].markdown("**Expected BF**")
            header_cols[4].markdown("**Orders**")
            header_cols[5].markdown("**Hist BF**")
            header_cols[6].markdown("**Avg Length**")
            header_cols[7].markdown("**Avg Width**")
            header_cols[8].markdown("**Thickness**")

            st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)

            # Display each KD material as a tree row
            for i, kd_item in enumerate(distribution):
                kd_mat = kd_item['KD_Material']
                contrib_pct = kd_item['Contribution_Pct']
                expected_bf = kd_item['Expected_BF_Output']
                order_count = kd_item['Order_Count']
                hist_bf = kd_item['Historical_BF_Output']

                # Parent row with expand toggle
                row_cols = st.columns([0.03, 0.15, 0.10, 0.12, 0.10, 0.12, 0.12, 0.12, 0.14])

                with row_cols[0]:
                    expanded = st.checkbox("", key=f"kd_expand_{i}", label_visibility="collapsed")

                arrow = "▼" if expanded else "▶"
                row_cols[1].markdown(f"{arrow} **{kd_mat}**")
                row_cols[2].markdown(f"**{contrib_pct:.1f}%**")
                row_cols[3].markdown(f"{expected_bf:,.0f}")
                row_cols[4].markdown(f"{order_count}")
                row_cols[5].markdown(f"{hist_bf:,.0f}")
                row_cols[6].markdown("-")
                row_cols[7].markdown("-")
                row_cols[8].markdown("-")

                # Child rows (grades) when expanded
                if expanded and has_grade_data:
                    # Use raw data if available, else fall back to precomputed
                    if has_raw_data:
                        grade_contributions = get_ks_grade_contributions_for_kd(
                            kd_material=kd_mat,
                            ks_material=result['ks_material'],
                            plant=result['plant'],
                            kd_contrib_pct=contrib_pct,
                            df_261_raw=df_261_raw,
                            df_101_raw=df_101_raw
                        )
                    else:
                        grade_contributions = get_ks_grade_contributions_for_kd_from_precomputed(
                            kd_material=kd_mat,
                            ks_material=result['ks_material'],
                            plant=result['plant'],
                            kd_contrib_pct=contrib_pct,
                            precomputed_data=precomputed_data
                        )

                    if grade_contributions:
                        for grade in grade_contributions:
                            grade_name = grade.get('Grade', 'N/A')
                            grade_pct = grade.get('Contribution_Pct', 0)
                            # Calculate expected BF for this grade based on parent's expected BF
                            # Formula: grade_expected_bf = parent_expected_bf × (grade_pct / parent_contrib_pct)
                            grade_expected_bf = expected_bf * (grade_pct / contrib_pct) if contrib_pct > 0 else 0
                            grade_orders = grade.get('Order_Count', 0)
                            avg_len = grade.get('Avg_Length', 0)
                            avg_wid = grade.get('Avg_Width', 0)
                            avg_thk = grade.get('Avg_Thickness', 0)

                            child_cols = st.columns([0.03, 0.15, 0.10, 0.12, 0.10, 0.12, 0.12, 0.12, 0.14])
                            child_cols[0].markdown("")
                            child_cols[1].markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└ {grade_name}")
                            child_cols[2].markdown(f"{grade_pct:.1f}%")
                            child_cols[3].markdown(f"{grade_expected_bf:,.0f}")
                            child_cols[4].markdown(f"{grade_orders}")
                            child_cols[5].markdown("-")
                            child_cols[6].markdown(f"{avg_len:.1f}" if avg_len else "-")
                            child_cols[7].markdown(f"{avg_wid:.1f}" if avg_wid else "-")
                            child_cols[8].markdown(f"{avg_thk:.1f}" if avg_thk else "-")

                        # Subtotal row for this KD material
                        total_grade_pct = sum(g.get('Contribution_Pct', 0) for g in grade_contributions)
                        subtotal_cols = st.columns([0.03, 0.15, 0.10, 0.12, 0.10, 0.12, 0.12, 0.12, 0.14])
                        subtotal_cols[0].markdown("")
                        subtotal_cols[1].markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Subtotal**")
                        subtotal_cols[2].markdown(f"**{total_grade_pct:.1f}%**")
                        subtotal_cols[3].markdown("")
                        subtotal_cols[4].markdown("")
                        subtotal_cols[5].markdown("")
                        subtotal_cols[6].markdown("")
                        subtotal_cols[7].markdown("")
                        subtotal_cols[8].markdown("")
                    else:
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;_No grade breakdown available_")
                elif expanded and not has_grade_data:
                    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;_No grade breakdown data available_")

            # Totals row
            total_pct = sum(d['Contribution_Pct'] for d in distribution)
            total_bf = sum(d['Expected_BF_Output'] for d in distribution)
            st.markdown(f"**Total: {total_pct:.1f}% | {total_bf:,.0f} BF**")

            # ================================================================
            # SUNBURST CHART - Visual representation of KD → Grade distribution
            # ================================================================
            if has_grade_data:
                st.markdown("---")
                st.subheader("Visual Distribution (KD Material → Grade)")

                # Build data for sunburst chart
                sunburst_data = []
                for kd_item in distribution:
                    kd_mat = kd_item['KD_Material']
                    kd_pct = kd_item['Contribution_Pct']
                    kd_bf = kd_item['Expected_BF_Output']

                    # Get grade breakdown for this KD material
                    if has_raw_data:
                        grades = get_ks_grade_contributions_for_kd(
                            kd_material=kd_mat,
                            ks_material=result['ks_material'],
                            plant=result['plant'],
                            kd_contrib_pct=kd_pct,
                            df_261_raw=df_261_raw,
                            df_101_raw=df_101_raw
                        )
                    else:
                        grades = get_ks_grade_contributions_for_kd_from_precomputed(
                            kd_material=kd_mat,
                            ks_material=result['ks_material'],
                            plant=result['plant'],
                            kd_contrib_pct=kd_pct,
                            precomputed_data=precomputed_data
                        )

                    if grades:
                        for g in grades:
                            grade_pct = g.get('Contribution_Pct', 0)
                            # Calculate expected BF for this grade
                            grade_expected_bf = kd_bf * (grade_pct / kd_pct) if kd_pct > 0 else 0
                            sunburst_data.append({
                                'KD_Material': kd_mat,
                                'Grade': g.get('Grade', 'Unknown'),
                                'Expected_BF': grade_expected_bf,
                                'Contribution_Pct': grade_pct
                            })
                    else:
                        # No grade breakdown - add KD material as single entry
                        sunburst_data.append({
                            'KD_Material': kd_mat,
                            'Grade': 'All Grades',
                            'Expected_BF': kd_bf,
                            'Contribution_Pct': kd_pct
                        })

                if sunburst_data:
                    sunburst_df = pd.DataFrame(sunburst_data)

                    # Create sunburst chart
                    fig = px.sunburst(
                        sunburst_df,
                        path=['KD_Material', 'Grade'],
                        values='Expected_BF',
                        title=f'Expected Output Distribution: {result["ks_material"]} → KD Materials → Grades',
                        color='Contribution_Pct',
                        color_continuous_scale='Blues',
                        hover_data={'Expected_BF': ':.0f', 'Contribution_Pct': ':.1f'}
                    )

                    fig.update_layout(
                        height=500,
                        margin=dict(t=50, l=0, r=0, b=0)
                    )

                    fig.update_traces(
                        textinfo='label+percent entry',
                        hovertemplate='<b>%{label}</b><br>Expected BF: %{value:,.0f}<br>Contribution: %{color:.1f}%<extra></extra>'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Inner ring: KD Materials | Outer ring: Grades | Click to zoom in")

            # Download button
            export_df = pd.DataFrame(distribution)
            export_df['KS_Material'] = result['ks_material']
            export_df['Plant'] = result['plant']
            export_df['Input_BF'] = result['input_bf']
            export_df['ML_Yield_Pct'] = result['ml_yield_pct']
            export_df['Historical_Yield_Pct'] = result['historical_yield_pct']
            export_df['Predicted_Total_Output_BF'] = result['predicted_output_bf']
            export_df['Prediction_Method'] = result.get('prediction_method', 'ML + Statistical')

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download KD Distribution (CSV)",
                data=csv,
                file_name=f"ml_prediction_{result['ks_material']}_{result['plant']}.csv",
                mime="text/csv"
            )

            # ================================================================
            # EXPLANATION SECTION - Hidden per user request
            # ================================================================
            # st.markdown("---")
            # with st.expander("How does this prediction work? (Click to learn more)", expanded=False):
            #     ... (explanation content hidden)


def get_historical_outputs_for_input(
    ks_material: str,
    plant: str,
    df_261_raw: Optional[pd.DataFrame] = None,
    df_101_raw: Optional[pd.DataFrame] = None,
    precomputed_data: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Get historically observed output materials for a KS material and plant."""
    # Preferred path: raw order-level data
    if df_261_raw is not None and df_101_raw is not None and len(df_261_raw) > 0 and len(df_101_raw) > 0:
        filtered_261 = df_261_raw[
            (df_261_raw['MATERIAL'] == ks_material) &
            (df_261_raw['PLANT'] == plant)
        ].copy()
        if len(filtered_261) > 0:
            order_list = filtered_261['MANUFACTURINGORDER'].unique()
            filtered_101 = df_101_raw[df_101_raw['MANUFACTURINGORDER'].isin(order_list)].copy()
            if len(filtered_101) > 0 and 'MATERIAL' in filtered_101.columns:
                return sorted([str(x) for x in filtered_101['MATERIAL'].dropna().unique()])

    # Fallback: precomputed KD distribution for KS+Plant
    if precomputed_data is not None:
        material_stats = precomputed_data.get('material_stats', {})
        ks_data = material_stats.get(ks_material, {})
        plant_data = ks_data.get(plant, {})
        kd_dist = plant_data.get('kd_distribution', [])
        if kd_dist:
            return sorted([str(kd.get('KD_Material')) for kd in kd_dist if kd.get('KD_Material')])

    return []


def render_test2_ml_only_section(
    options,
    model,
    classifier,
    encoders,
    feature_columns,
    classifier_features,
    classifier_source="missing",
    df_261_raw=None,
    df_101_raw=None,
    precomputed_data=None,
    kd_model_artifacts=None,
):
    """Render V2.1: ML-only KD material distribution prediction."""
    from src.kd_material_prediction import predict_kd_distribution

    use_kd_model = kd_model_artifacts is not None

    st.subheader("V2.1: ML Output Material Distribution")
    if use_kd_model:
        st.caption("Material-Aware KNN + XGBoost: predicts yield and output material distribution from historical patterns")
    else:
        st.warning("KD distribution model not found. Using legacy classifier approach (less accurate).")

    # --- Input Mode Toggle ---
    input_mode = st.radio(
        "Input Mode",
        ["Manual Entry", "Upload 261 File"],
        horizontal=True,
        key="t2_input_mode",
    )

    limit_historical = True
    exclude_other = False

    if input_mode == "Manual Entry":
        # ---- MANUAL ENTRY MODE ----
        col1, col2 = st.columns(2)

        with col1:
            plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
            selected_plant = st.selectbox("Plant *", options=plant_options, key="t2_plant")

            material_options = options.get('Input_Material', [])
            mat_search = st.text_input(
                "Search Material",
                key="t2_mat_search",
                placeholder="Type to filter..."
            )
            if mat_search:
                filtered_materials = [m for m in material_options if mat_search.upper() in str(m).upper()]
            else:
                filtered_materials = material_options

            selected_material = st.selectbox(
                f"Material * ({len(filtered_materials)} available)",
                options=filtered_materials if filtered_materials else ['No materials found'],
                key="t2_material"
            )

            specie_options = options.get('Input_Specie', ['SM'])
            selected_specie = st.selectbox("Specie *", options=specie_options, key="t2_specie")

            grade_options = options.get('Input_Grade', ['2C'])
            selected_grade = st.selectbox("Grade *", options=grade_options, key="t2_grade")

        with col2:
            input_bf = st.number_input(
                "Input BF (BFIN) *",
                min_value=0.0,
                max_value=10000000.0,
                value=10000.0,
                step=1000.0,
                key="t2_bfin"
            )
            input_thickness = st.number_input(
                "Thickness *",
                min_value=0.0,
                max_value=100.0,
                value=4.0,
                step=0.1,
                key="t2_thickness"
            )
            input_width = st.number_input(
                "Width *",
                min_value=0.0,
                max_value=100.0,
                value=8.0,
                step=0.1,
                key="t2_width"
            )
            limit_historical = st.checkbox(
                "Limit to historically seen outputs for this input material",
                value=True,
                key="t2_hist_only"
            )
            if use_kd_model:
                exclude_other = st.checkbox(
                    "Redistribute 'Other' across named materials",
                    value=False,
                    key="t2_exclude_other",
                    help="Removes the catch-all 'Other Materials' bucket and spreads its BF across the named KD materials."
                )

            # Legacy classifier controls
            if not use_kd_model:
                top_k = st.slider("Top related outputs", min_value=3, max_value=10, value=5, step=1, key="t2_top_k")
                confidence_threshold_pct = st.slider(
                    "Minimum top probability (%)",
                    min_value=10, max_value=80, value=35, step=5, key="t2_conf_thr"
                )
                kd_only = st.checkbox("Show KD materials only", value=True, key="t2_kd_only")

        st.markdown("---")

        if st.button("Predict Output Distribution", type="primary", use_container_width=True, key="t2_run"):
            if selected_material == 'No materials found':
                st.error("Please select a valid material.")
                return

            input_data = {
                'Input_Plant': selected_plant,
                'Input_Material': selected_material,
                'Input_Thickness': input_thickness,
                'Input_Specie': selected_specie,
                'Input_Grade': selected_grade,
                'Input_Width': input_width,
                'Total_Input_BF': input_bf,
                'Input_Length': 96.0,
            }

            with st.spinner("Running ML prediction..."):
                if use_kd_model:
                    kd_bundle = kd_model_artifacts['model_bundle']
                    kd_template = kd_model_artifacts['template_df']
                    kd_cols = [c for c in kd_template.columns if c.startswith("KD_")]
                    material_history = kd_model_artifacts.get('material_history')

                    result_df, summary = predict_kd_distribution(
                        input_data=input_data,
                        model_bundle=kd_bundle,
                        template_df=kd_template,
                        kd_cols=kd_cols,
                        material_history=material_history,
                        apply_material_guardrail=limit_historical,
                        exclude_other=exclude_other,
                    )

                    st.session_state.t2_result = {
                        'mode': 'kd_distribution',
                        'input_data': input_data,
                        'result_df': result_df,
                        'summary': summary,
                    }
                    st.session_state.t2_original_result_df = result_df.copy()
                    st.session_state.t2_exclude_selection = []
                else:
                    if model is None:
                        st.error("Yield model not available.")
                        return
                    if classifier is None:
                        st.error("Output classifier not available.")
                        return

                    yield_result = forward_predict(model, input_data, encoders, feature_columns)
                    output_preds = predict_output_material(
                        classifier=classifier,
                        input_data=input_data,
                        encoders=encoders,
                        classifier_features=classifier_features,
                        top_k=top_k
                    )
                    if kd_only:
                        output_preds = [p for p in output_preds if 'KD' in str(p.get('output_material', '')).upper()]

                    historical_outputs = get_historical_outputs_for_input(
                        ks_material=selected_material,
                        plant=selected_plant,
                        df_261_raw=df_261_raw,
                        df_101_raw=df_101_raw,
                        precomputed_data=precomputed_data
                    )
                    if limit_historical and historical_outputs:
                        historical_set = set(historical_outputs)
                        output_preds = [p for p in output_preds if str(p.get('output_material', '')) in historical_set]

                    total_output_bf = float(yield_result.get('predicted_output_bf', 0))
                    total_prob = sum(p.get('probability', 0.0) for p in output_preds)
                    norm_denominator = total_prob if total_prob > 0 else 1.0
                    alloc_rows = []
                    for p in output_preds:
                        prob = float(p.get('probability', 0.0))
                        alloc_rows.append({
                            'Output Material': p.get('output_material'),
                            'Distribution %': round(prob / norm_denominator * 100, 2),
                            'Expected Output BF': round(total_output_bf * (prob / norm_denominator), 2),
                        })
                    result_df = pd.DataFrame(alloc_rows)
                    summary = {
                        'predicted_yield_pct': yield_result.get('predicted_yield_pct', 0),
                        'predicted_output': total_output_bf,
                        'kd_count': len(result_df),
                        'total_bfin': input_bf,
                        'yield_factor': yield_result.get('predicted_yield_pct', 0) / 100,
                    }

                    st.session_state.t2_result = {
                        'mode': 'legacy',
                        'input_data': input_data,
                        'result_df': result_df,
                        'summary': summary,
                    }

    else:
        # ---- UPLOAD 261 FILE MODE ----
        if not use_kd_model:
            st.error("261 file upload requires the KD distribution model. Model not found.")
            return

        uploaded_file = st.file_uploader(
            "Upload your 261 (Goods Issue) CSV file:",
            type=["csv"],
            key="t2_file_upload",
            help="CSV should contain 261 consumption records with columns: MANUFACTURINGORDER, MATERIAL, BFIN, MATERIALSPECIE, MATERIALTHICKNESS, PLANT, TALLYGRADE, TALLYWIDTH"
        )

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            df_upload.columns = df_upload.columns.str.upper().str.strip()

            # Filter to 261 movement type if column exists
            if "GOODSMOVEMENTTYPE" in df_upload.columns:
                df_upload = df_upload[df_upload["GOODSMOVEMENTTYPE"].astype(str) == "261"].copy()

            # Validate required columns
            required_cols = ["MANUFACTURINGORDER", "MATERIAL", "BFIN"]
            recommended_cols = ["MATERIALSPECIE", "MATERIALTHICKNESS", "PLANT", "TALLYGRADE", "TALLYWIDTH"]

            missing_required = [c for c in required_cols if c not in df_upload.columns]
            missing_recommended = [c for c in recommended_cols if c not in df_upload.columns]

            if missing_required:
                st.error(f"Missing required columns: {', '.join(missing_required)}")
                return

            if missing_recommended:
                st.warning(f"Missing optional columns (predictions may be less accurate): {', '.join(missing_recommended)}")

            # Ensure numeric types
            df_upload["BFIN"] = pd.to_numeric(df_upload["BFIN"], errors="coerce").fillna(0)

            # Aggregate per manufacturing order
            order_agg = df_upload.groupby("MANUFACTURINGORDER").agg(
                Material=("MATERIAL", "first"),
                Total_BFIN=("BFIN", "sum"),
                Rows=("BFIN", "count"),
                Specie=("MATERIALSPECIE", "first") if "MATERIALSPECIE" in df_upload.columns else ("MATERIAL", lambda x: ""),
                Thickness=("MATERIALTHICKNESS", "first") if "MATERIALTHICKNESS" in df_upload.columns else ("MATERIAL", lambda x: 0),
                Plant=("PLANT", "first") if "PLANT" in df_upload.columns else ("MATERIAL", lambda x: ""),
                Grade=("TALLYGRADE", "first") if "TALLYGRADE" in df_upload.columns else ("MATERIAL", lambda x: ""),
                Width=("TALLYWIDTH", "max") if "TALLYWIDTH" in df_upload.columns else ("MATERIAL", lambda x: 0),
            ).reset_index()

            st.success(f"Loaded **{len(df_upload):,}** rows with **{len(order_agg)}** manufacturing orders")

            # Preview table
            preview_df = order_agg[["MANUFACTURINGORDER", "Material", "Total_BFIN", "Rows"]].copy()
            preview_df.columns = ["Order", "Input Material", "Total Input BF", "Rows"]
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            # Options
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                limit_historical = st.checkbox(
                    "Limit to historically seen outputs",
                    value=True,
                    key="t2_upload_hist_only"
                )
            with col_opt2:
                exclude_other = st.checkbox(
                    "Redistribute 'Other' across named materials",
                    value=False,
                    key="t2_upload_exclude_other",
                )

            st.markdown("---")

            if st.button("Predict All Orders", type="primary", use_container_width=True, key="t2_upload_run"):
                kd_bundle = kd_model_artifacts['model_bundle']
                kd_template = kd_model_artifacts['template_df']
                kd_cols = [c for c in kd_template.columns if c.startswith("KD_")]
                material_history = kd_model_artifacts.get('material_history')

                all_results = []
                all_summaries = []
                progress_bar = st.progress(0)

                for i, row in order_agg.iterrows():
                    order_id = row["MANUFACTURINGORDER"]
                    input_data = {
                        'Input_Plant': str(row.get("Plant", "")),
                        'Input_Material': str(row["Material"]),
                        'Input_Thickness': float(row.get("Thickness", 0)) if pd.notna(row.get("Thickness")) else 0.0,
                        'Input_Specie': str(row.get("Specie", "")),
                        'Input_Grade': str(row.get("Grade", "")),
                        'Input_Width': float(row.get("Width", 0)) if pd.notna(row.get("Width")) else 0.0,
                        'Total_Input_BF': float(row["Total_BFIN"]),
                        'Input_Length': 96.0,
                    }

                    # Get full 261 rows for this order (all lengths/widths/grades)
                    order_rows = df_upload[df_upload["MANUFACTURINGORDER"] == order_id]

                    try:
                        result_df, summary = predict_kd_distribution(
                            input_data=input_data,
                            model_bundle=kd_bundle,
                            template_df=kd_template,
                            kd_cols=kd_cols,
                            material_history=material_history,
                            apply_material_guardrail=limit_historical,
                            exclude_other=exclude_other,
                            df_261_rows=order_rows,
                        )
                        result_df.insert(0, "Order", order_id)
                        result_df.insert(1, "Input Material", str(row["Material"]))
                        all_results.append(result_df)
                        all_summaries.append({
                            'order': order_id,
                            'input_material': str(row["Material"]),
                            **summary,
                        })
                    except Exception as e:
                        st.warning(f"Order {order_id}: prediction failed - {e}")

                    progress_bar.progress((i + 1) / len(order_agg))

                progress_bar.empty()

                if all_results:
                    combined_df = pd.concat(all_results, ignore_index=True)
                    summary_df = pd.DataFrame(all_summaries)

                    # Store in session state for display
                    st.session_state.t2_result = {
                        'mode': 'kd_distribution_batch',
                        'input_data': {'Input_Material': 'Batch Upload', 'Input_Plant': 'Multiple'},
                        'result_df': combined_df,
                        'summary': {
                            'predicted_yield_pct': round(summary_df['predicted_yield_pct'].mean(), 2),
                            'predicted_output': round(summary_df['predicted_output'].sum(), 2),
                            'kd_count': combined_df['Output Material'].nunique() if 'Output Material' in combined_df.columns else 0,
                            'total_bfin': round(summary_df['total_bfin'].sum(), 2),
                            'n_orders': len(summary_df),
                        },
                        'summary_df': summary_df,
                    }
                else:
                    st.error("No predictions could be generated.")

    # --- Display results ---
    if st.session_state.get('t2_result'):
        result = st.session_state['t2_result']
        input_data = result['input_data']
        result_df = result['result_df']
        summary = result['summary']
        is_batch = result.get('mode') == 'kd_distribution_batch'

        st.markdown("---")
        st.header("Prediction Results")

        if result.get('mode') in ('kd_distribution', 'kd_distribution_batch'):
            if is_batch:
                n_orders = summary.get('n_orders', 0)
                st.info(f"Prediction Method: Material-Aware KNN + XGBoost Yield | **{n_orders} orders** processed")
            else:
                search_mode = summary.get('search_mode', 'unknown')
                n_nbrs = summary.get('n_neighbors_used', '?')
                avg_dist = summary.get('avg_neighbor_distance', '?')
                st.info(
                    f"Prediction Method: Material-Aware KNN + XGBoost Yield  \n"
                    f"Search: **{search_mode}** | Neighbors: **{n_nbrs}** | Avg distance: **{avg_dist}**"
                )
        else:
            st.warning("Prediction Method: Legacy Classifier (probabilities are confidence scores, not production proportions)")

        st.caption(
            "Finds the most similar historical orders for this input material and averages their actual output distributions."
        )

        if is_batch:
            col_a, col_b, col_c, col_d = st.columns(4)
        else:
            col_a, col_b, col_c = st.columns(3)
        with col_a:
            label = "Avg Yield %" if is_batch else "Overall Yield %"
            st.metric(label, f"{summary.get('predicted_yield_pct', 0):.1f}%")
        with col_b:
            st.metric("Total Expected Output BF", f"{summary.get('predicted_output', 0):,.0f}")
        with col_c:
            st.metric("Output Materials Found", f"{summary.get('kd_count', 0)}")
        if is_batch:
            with col_d:
                st.metric("Orders Processed", f"{summary.get('n_orders', 0)}")

        if result_df is None or result_df.empty:
            st.warning("No output materials predicted for this input combination.")
            return

        # Batch: show per-order summary table first
        if is_batch and 'summary_df' in result:
            st.subheader("Per-Order Summary")
            sum_df = result['summary_df'][['order', 'input_material', 'total_bfin', 'predicted_yield_pct', 'predicted_output', 'kd_count']].copy()
            sum_df.columns = ['Order', 'Input Material', 'Input BF', 'Yield %', 'Expected Output BF', 'Output Materials']
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

        st.subheader("Output Material Distribution")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # --- Material Exclusion (single prediction only) ---
        if not is_batch:
            orig_df = st.session_state.get("t2_original_result_df")
            total_output = summary.get("predicted_output", 0)
            total_bfin_val = summary.get("total_bfin", 0)

            if orig_df is not None and len(orig_df) > 0:
                st.markdown("---")
                st.subheader("Exclude Materials")
                exclude_sel = st.multiselect(
                    "Select materials to exclude (their BF redistributed equally among remaining):",
                    options=orig_df["Output Material"].tolist(),
                    default=st.session_state.get("t2_exclude_selection", []),
                    key="t2_exclude_widget",
                )
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    apply_clicked = st.button(
                        "Apply Exclusion", key="t2_apply_excl",
                        type="primary", use_container_width=True
                    )
                with col_ex2:
                    reset_clicked = st.button(
                        "Reset to Original", key="t2_reset_excl",
                        use_container_width=True
                    )

                if apply_clicked and exclude_sel:
                    st.session_state.t2_exclude_selection = exclude_sel
                    excl_mask = orig_df["Output Material"].isin(exclude_sel)
                    excluded_share = orig_df.loc[excl_mask, "Distribution %"].sum()
                    remaining = orig_df[~excl_mask].copy()
                    n = len(remaining)
                    if n > 0:
                        per_mat_boost = excluded_share / n
                        remaining["Distribution %"] = (remaining["Distribution %"] + per_mat_boost).round(2)
                        remaining["Expected Output BF"] = ((remaining["Distribution %"] / 100) * total_output).round(2)
                        remaining["Material Yield %"] = (
                            (remaining["Expected Output BF"] / total_bfin_val * 100).round(2)
                            if total_bfin_val > 0 else 0.0
                        )
                        remaining = remaining.sort_values("Expected Output BF", ascending=False).reset_index(drop=True)
                        remaining["Cumulative %"] = remaining["Distribution %"].cumsum().round(2)
                        st.session_state.t2_result["result_df"] = remaining
                        st.session_state.t2_result["summary"]["kd_count"] = len(remaining)
                        st.rerun()

                if reset_clicked:
                    st.session_state.t2_exclude_selection = []
                    st.session_state.t2_result["result_df"] = orig_df.copy()
                    st.session_state.t2_result["summary"]["kd_count"] = len(orig_df)
                    st.rerun()

        # Export
        export_df = result_df.copy()
        if not is_batch:
            export_df['Input Material'] = input_data.get('Input_Material')
            export_df['Plant'] = input_data.get('Input_Plant')
            export_df['Input Thickness'] = input_data.get('Input_Thickness')
            export_df['Input Width'] = input_data.get('Input_Width')
            export_df['Input Specie'] = input_data.get('Input_Specie')
            export_df['Input Grade'] = input_data.get('Input_Grade')
        export_df['Overall Yield %'] = summary.get('predicted_yield_pct', 0)
        export_df['Total Expected Output BF'] = summary.get('predicted_output', 0)

        csv = export_df.to_csv(index=False)
        file_label = "batch" if is_batch else input_data.get('Input_Material', 'material')
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name=f"kd_distribution_{file_label}.csv",
            mime="text/csv",
            key="t2_download"
        )


def apply_v22_distribution_rules(result_df: pd.DataFrame) -> pd.DataFrame:
    """Apply NKD→KD merge and 3BKD exclusion to V2.2 nested distribution data."""
    new_rows = []
    for _, row in result_df.iterrows():
        row_dict = row.to_dict()
        dist = row_dict.get("distribution", [])
        if not isinstance(dist, list):
            new_rows.append(row_dict)
            continue

        # Step 1: NKD → KD merge (transfer bfout to KD counterpart)
        bf_map = {}
        for d in dist:
            if not isinstance(d, dict):
                continue
            mat = d.get("material", "")
            bf_map[mat] = bf_map.get(mat, 0) + d.get("bfout", 0)

        merged = {}
        for mat, bf in bf_map.items():
            if mat.endswith("NKD"):
                target = mat[:-3] + "KD"
                merged[target] = merged.get(target, 0) + bf
            else:
                merged[mat] = merged.get(mat, 0) + bf

        # Step 2: 3BKD exclusion with proportional redistribution
        excluded_bf = sum(bf for mat, bf in merged.items() if "3BKD" in mat)
        kept = {mat: bf for mat, bf in merged.items() if "3BKD" not in mat}
        remaining_total = sum(kept.values())
        if excluded_bf > 0 and remaining_total > 0:
            scale = (remaining_total + excluded_bf) / remaining_total
            kept = {mat: round(bf * scale, 2) for mat, bf in kept.items()}

        row_dict["distribution"] = [{"material": mat, "bfout": bf} for mat, bf in kept.items()]
        row_dict["kd_count"] = len(kept)
        new_rows.append(row_dict)
    return pd.DataFrame(new_rows)


def render_final_v2_section():
    """Render the V2.2 section - External API prediction with configurable options."""
    st.subheader("V2.2: External API Prediction")
    st.caption("Upload 261 consumption data and call external API for material distribution predictions")

    API_ENDPOINT = "https://predictkd-production.up.railway.app/predict"

    # --- Step 1: File Upload ---
    uploaded_file = st.file_uploader(
        "Upload your 261 (Goods Issue) CSV file:",
        type=["csv"],
        key="fv2_file_upload",
        help="CSV should contain 261 consumption records with columns: MANUFACTURINGORDER, MATERIAL, BFIN, MATERIALSPECIE, MATERIALTHICKNESS, PLANT, TALLYGRADE, TALLYLENGTH, TALLYWIDTH"
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to continue.")
        return

    try:
        df_upload = pd.read_csv(uploaded_file)
        df_upload.columns = df_upload.columns.str.upper().str.strip()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    # Filter to 261 movement type if column exists
    if "GOODSMOVEMENTTYPE" in df_upload.columns:
        original_count = len(df_upload)
        df_upload = df_upload[df_upload["GOODSMOVEMENTTYPE"].astype(str) == "261"].copy()
        if len(df_upload) < original_count:
            st.info(f"Filtered to {len(df_upload):,} rows with GOODSMOVEMENTTYPE=261 (from {original_count:,} total)")

    # Validate required columns
    required_cols = [
        "MANUFACTURINGORDER", "MATERIAL", "BFIN", "MATERIALSPECIE",
        "MATERIALTHICKNESS", "PLANT", "TALLYGRADE", "TALLYLENGTH", "TALLYWIDTH"
    ]
    missing_cols = [col for col in required_cols if col not in df_upload.columns]

    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.caption(f"Available columns: {', '.join(df_upload.columns.tolist())}")
        return

    # Ensure numeric types
    df_upload["BFIN"] = pd.to_numeric(df_upload["BFIN"], errors="coerce").fillna(0)
    df_upload["MATERIALTHICKNESS"] = pd.to_numeric(df_upload["MATERIALTHICKNESS"], errors="coerce").fillna(0)
    df_upload["TALLYLENGTH"] = pd.to_numeric(df_upload["TALLYLENGTH"], errors="coerce").fillna(0)
    df_upload["TALLYWIDTH"] = pd.to_numeric(df_upload["TALLYWIDTH"], errors="coerce").fillna(0)

    # Show data preview
    unique_orders = df_upload["MANUFACTURINGORDER"].nunique()
    unique_materials = df_upload["MATERIAL"].nunique()
    total_bfin = df_upload["BFIN"].sum()

    st.success(f"Loaded **{len(df_upload):,}** rows | **{unique_orders}** orders | **{unique_materials}** unique materials | Total BFIN: **{total_bfin:,.0f}**")
    st.dataframe(df_upload.head(10), use_container_width=True, hide_index=True)

    # --- Step 2: Output Level Selection ---
    st.markdown("---")
    output_level_option = st.radio(
        "Select output level:",
        ["Material", "Detailed"],
        horizontal=True,
        key="fv2_output_level",
        help="Material: Material-level aggregation. Detailed: Board-level detail."
    )

    output_level_value = "detailed" if output_level_option == "Detailed" else "material"

    # Exclude materials — populated from API response (shown after first API call)
    exclude_materials = st.session_state.get('fv2_exclude_selection', [])

    # --- Step 3: Configurable Options ---
    st.markdown("---")
    top_n = st.number_input(
        "Top N Results",
        min_value=1,
        max_value=100,
        value=15,
        step=1,
        key="fv2_top_n",
        help="Number of top results to return from API"
    )

    # --- Step 4: API Call ---
    st.markdown("---")
    if st.button("Call External API", type="primary", use_container_width=True, key="fv2_predict_btn"):
        today = datetime.now().strftime("%Y-%m-%d")

        # Build inputs array from CSV rows
        inputs = []
        for _, row in df_upload.iterrows():
            inputs.append({
                "material": str(row["MATERIAL"]),
                "species": str(row["MATERIALSPECIE"]),
                "thickness": float(row["MATERIALTHICKNESS"]),
                "plant": str(row["PLANT"]),
                "tallygrade": str(row["TALLYGRADE"]),
                "tallylength": float(row["TALLYLENGTH"]),
                "tallywidth": float(row["TALLYWIDTH"]),
                "bfin": float(row["BFIN"]),
                "posting_date": today,
                "manufacturing_order": str(row["MANUFACTURINGORDER"])
            })

        # Build options object
        api_options = {
            "output_level": output_level_value,
            "exclude_materials": exclude_materials,
            "top_n": int(top_n)
        }

        request_body = {
            "inputs": inputs,
            "options": api_options
        }

        st.caption(f"Sending {len(inputs)} input rows | Output level: {output_level_value} | Top N: {top_n}")

        progress_bar = st.progress(0, text="Sending request to API...")

        try:
            response = requests.post(
                API_ENDPOINT,
                json=request_body,
                timeout=300,
                headers={"Content-Type": "application/json"}
            )
            progress_bar.progress(50, text="Processing API response...")

            if response.status_code == 200:
                api_response = response.json()
                progress_bar.progress(100, text="Success!")
                progress_bar.empty()

                st.session_state.fv2_response = api_response
                st.session_state.fv2_inputs = inputs  # Store for re-call with exclusions
                st.session_state.fv2_request_info = {
                    "rows": len(inputs),
                    "output_level": output_level_value,
                    "top_n": top_n,
                    "exclude_count": len(exclude_materials),
                    "timestamp": today
                }

                # Extract KD materials from response for exclude dropdown
                kd_materials_from_response = set()
                try:
                    # Parse response into a temp DataFrame to find KD material values
                    temp_df = None
                    if isinstance(api_response, dict):
                        for key in ('predictions', 'results', 'data'):
                            if key in api_response and isinstance(api_response[key], list):
                                temp_df = pd.DataFrame(api_response[key])
                                # Check nested distribution arrays
                                for item in api_response[key]:
                                    if isinstance(item, dict):
                                        for dist in item.get('distribution', []):
                                            if 'output_material' in dist:
                                                kd_materials_from_response.add(str(dist['output_material']))
                                        if 'output_material' in item:
                                            kd_materials_from_response.add(str(item['output_material']))
                                break
                    elif isinstance(api_response, list):
                        temp_df = pd.DataFrame(api_response)

                    # Also check DataFrame columns for KD material values
                    if temp_df is not None and not temp_df.empty:
                        for col in ['output_material', 'material', 'kd_material', 'MATERIAL', 'OUTPUT_MATERIAL']:
                            if col in temp_df.columns:
                                vals = temp_df[col].dropna().astype(str).unique()
                                kd_vals = [v for v in vals if 'KD' in v.upper()]
                                kd_materials_from_response.update(kd_vals)
                except Exception:
                    pass
                st.session_state.fv2_kd_materials = sorted(kd_materials_from_response)

                st.success("API call successful!")
            else:
                progress_bar.empty()
                st.error(f"API returned status {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            progress_bar.empty()
            st.error("API request timed out (5 minute limit). The server may be processing a large request.")
        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            st.error(f"Connection error. Unable to reach: {API_ENDPOINT}")
        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            st.error(f"Request error: {str(e)}")

    # --- Step 5: Display Results ---
    if st.session_state.get('fv2_response') is not None:
        api_response = st.session_state.fv2_response
        request_info = st.session_state.get('fv2_request_info', {})

        st.markdown("---")
        st.header("Prediction Results")
        st.caption(
            f"Rows: {request_info.get('rows', 'N/A')} | "
            f"Output level: {request_info.get('output_level', 'N/A')} | "
            f"Top N: {request_info.get('top_n', 'N/A')} | "
            f"Excluded: {request_info.get('exclude_count', 0)}"
        )

        # Parse response into DataFrame
        result_df = None
        if isinstance(api_response, dict):
            for key in ('predictions', 'results', 'data'):
                if key in api_response:
                    result_df = pd.DataFrame(api_response[key])
                    break
            if result_df is None:
                result_df = pd.DataFrame([api_response])
        elif isinstance(api_response, list):
            result_df = pd.DataFrame(api_response)

        # Apply NKD→KD merge and 3BKD exclusion to distribution data
        if result_df is not None and not result_df.empty and "distribution" in result_df.columns:
            result_df = apply_v22_distribution_rules(result_df)

        if result_df is not None and not result_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Results", f"{len(result_df):,}")
            with col2:
                st.metric("Input Rows Sent", f"{request_info.get('rows', 0):,}")
            with col3:
                st.metric("Request Date", request_info.get('timestamp', 'N/A'))

            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # --- Exclude Materials Dropdown (populated from API response) ---
            # Extract KD materials directly from the result DataFrame
            kd_materials_available = []
            if 'material' in result_df.columns:
                kd_materials_available = sorted(result_df['material'].dropna().astype(str).unique().tolist())
            elif 'output_material' in result_df.columns:
                kd_materials_available = sorted(result_df['output_material'].dropna().astype(str).unique().tolist())

            if kd_materials_available:
                st.markdown("---")
                st.subheader("Exclude KD Materials")
                exclude_selection = st.multiselect(
                    "Select KD materials to exclude, then click 'Call External API' again:",
                    options=kd_materials_available,
                    default=[],
                    key="fv2_exclude_materials",
                    help="These KD output materials will be excluded from the next API call."
                )
                st.session_state.fv2_exclude_selection = exclude_selection
                if exclude_selection:
                    st.info(f"Excluding {len(exclude_selection)} material(s): {', '.join(exclude_selection)}")

                    # Re-call API with exclusions button
                    if st.button("Re-call API with Exclusions", type="primary", use_container_width=True, key="fv2_exclude_btn"):
                        stored_inputs = st.session_state.get('fv2_inputs', [])
                        if stored_inputs:
                            re_request_info = st.session_state.get('fv2_request_info', {})
                            api_options = {
                                "output_level": re_request_info.get('output_level', 'material'),
                                "exclude_materials": exclude_selection,
                                "top_n": int(re_request_info.get('top_n', 15))
                            }
                            request_body = {
                                "inputs": stored_inputs,
                                "options": api_options
                            }

                            exclude_progress = st.progress(0, text="Re-calling API with exclusions...")
                            try:
                                re_response = requests.post(
                                    "https://predictkd-production.up.railway.app/predict",
                                    json=request_body,
                                    timeout=300,
                                    headers={"Content-Type": "application/json"}
                                )
                                exclude_progress.progress(50, text="Processing response...")

                                if re_response.status_code == 200:
                                    new_response = re_response.json()
                                    exclude_progress.progress(100, text="Success!")
                                    exclude_progress.empty()

                                    st.session_state.fv2_response = new_response
                                    st.session_state.fv2_request_info = {
                                        "rows": len(stored_inputs),
                                        "output_level": re_request_info.get('output_level', 'material'),
                                        "top_n": re_request_info.get('top_n', 15),
                                        "exclude_count": len(exclude_selection),
                                        "timestamp": re_request_info.get('timestamp', '')
                                    }
                                    st.rerun()
                                else:
                                    exclude_progress.empty()
                                    st.error(f"API returned status {re_response.status_code}: {re_response.text}")
                            except Exception as e:
                                exclude_progress.empty()
                                st.error(f"Request error: {str(e)}")
                        else:
                            st.warning("No stored input data. Please call the API first using the button above.")

            csv_output = result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name=f"final_v2_predictions_{request_info.get('timestamp', 'export')}.csv",
                mime="text/csv",
                key="fv2_download",
                use_container_width=True
            )
        else:
            st.warning("No prediction results to display.")

        with st.expander("View Raw API Response"):
            st.json(api_response)


def render_kd_material_lookup_section(options, historical_data, df_261_raw=None, df_101_raw=None, precomputed_data=None):
    """Render the KD Material Lookup section - find historical KD outputs for a KS input.

    Args:
        options: Dropdown options dict
        historical_data: Joined historical data (for distribution calculation)
        df_261_raw: RAW 261 CSV data (for accurate BFIN totals matching notebook)
        df_101_raw: RAW 101 CSV data (for accurate BFOUT totals matching notebook)
        precomputed_data: Pre-computed statistics dict (for deployment mode)
    """
    st.subheader("KD Material Lookup: Find Output Materials")
    st.caption("Enter a KS (input) material to see which KD (output) materials it has historically produced")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Material (KS)**")

        # Plant selection
        plant_options = options.get('Input_Plant', ['1M02', '1Y01'])
        kd_plant = st.selectbox(
            "Plant",
            options=plant_options if plant_options else ['1M02'],
            key="kd_plant"
        )

        # Material selection with search
        material_options = options.get('Input_Material', [])
        kd_mat_search = st.text_input(
            "Search Material",
            key="kd_mat_search",
            placeholder="Type to filter (e.g., 4PO3BKS)..."
        )

        if kd_mat_search:
            filtered = [m for m in material_options if kd_mat_search.upper() in str(m).upper()]
        else:
            filtered = material_options

        kd_material = st.selectbox(
            f"Input Material ({len(filtered)} available)",
            options=filtered if filtered else ['No materials found'],
            key="kd_material"
        )

    with col2:
        st.markdown("**Material Properties (Optional Filters)**")

        # Thickness
        thickness_options = options.get('Input_Thickness', [4, 6, 8])
        kd_thickness = st.selectbox(
            "Thickness",
            options=[None] + list(thickness_options) if thickness_options else [None, 4, 6, 8],
            format_func=lambda x: "Any" if x is None else str(x),
            key="kd_thickness"
        )

        # Species
        specie_options = options.get('Input_Specie', ['SM', 'AS', 'WO'])
        kd_specie = st.selectbox(
            "Species",
            options=[None] + list(specie_options) if specie_options else [None],
            format_func=lambda x: "Any" if x is None else str(x),
            key="kd_specie"
        )

        # Grade
        grade_options = options.get('Input_Grade', ['2C', '1C', '3A'])
        kd_grade = st.selectbox(
            "Grade",
            options=[None] + list(grade_options) if grade_options else [None],
            format_func=lambda x: "Any" if x is None else str(x),
            key="kd_grade"
        )

    st.markdown("---")

    # Quantity and Wastage inputs
    col3, col4 = st.columns(2)

    with col3:
        kd_input_bf = st.number_input(
            "Input Quantity (BF)",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            key="kd_input_bf",
            help="Total board feet of input material"
        )

    with col4:
        kd_wastage = st.number_input(
            "Wastage %",
            min_value=0.0,
            max_value=50.0,
            value=9.0,
            step=0.5,
            key="kd_wastage",
            help="Percentage of material lost during production (default 9%)"
        )

    # Find KD Materials button
    if st.button("Find KD Materials", type="primary", key="kd_lookup_btn", use_container_width=True):
        # Check available data sources
        has_historical = historical_data is not None and len(historical_data) > 0
        has_precomputed_kd = precomputed_data is not None and 'kd_lookup' in precomputed_data

        if not has_historical and not has_precomputed_kd:
            st.error("No historical data available. Please ensure data files are loaded or pre-computed data exists.")
            return

        if kd_material == 'No materials found':
            st.error("Please select a valid input material.")
            return

        with st.spinner("Looking up historical KD outputs..."):
            # Get historical distribution - try precomputed first, then historical
            if has_precomputed_kd:
                kd_distribution = get_historical_kd_distribution_from_precomputed(
                    input_material=kd_material,
                    kd_lookup=precomputed_data['kd_lookup'],
                    input_thickness=kd_thickness,
                    input_grade=kd_grade,
                    input_species=kd_specie,
                    input_plant=kd_plant
                )
            else:
                kd_distribution = get_historical_kd_distribution(
                    input_material=kd_material,
                    historical_data=historical_data,
                    input_thickness=kd_thickness,
                    input_grade=kd_grade,
                    input_species=kd_specie,
                    input_plant=kd_plant
                )

            if not kd_distribution:
                st.warning(f"No historical data found for material: **{kd_material}**")
                st.info("Try removing the optional filters (Thickness, Species, Grade) or selecting a different material.")
                return

            # Calculate total historical BF Input/Output
            total_hist_input_bf = 0
            total_hist_output_bf = 0

            # Option 1: Use RAW CSV data for accurate totals
            if df_261_raw is not None and df_101_raw is not None:
                # Use RAW 261 data for BFIN (column names: MATERIAL, BFIN, PLANT, etc.)
                filtered_261 = df_261_raw[df_261_raw['MATERIAL'] == kd_material].copy()
                if kd_thickness is not None and 'MATERIALTHICKNESS' in filtered_261.columns:
                    filtered_261 = filtered_261[filtered_261['MATERIALTHICKNESS'] == kd_thickness]
                if kd_specie is not None and 'MATERIALSPECIE' in filtered_261.columns:
                    filtered_261 = filtered_261[filtered_261['MATERIALSPECIE'] == kd_specie]
                if kd_grade is not None and 'TALLYGRADE' in filtered_261.columns:
                    filtered_261 = filtered_261[filtered_261['TALLYGRADE'] == kd_grade]
                if kd_plant is not None and 'PLANT' in filtered_261.columns:
                    filtered_261 = filtered_261[filtered_261['PLANT'] == kd_plant]
                total_hist_input_bf = filtered_261['BFIN'].sum() if 'BFIN' in filtered_261.columns else 0

                # Get MANUFACTURINGORDER list from filtered 261 data
                order_list = filtered_261['MANUFACTURINGORDER'].unique() if 'MANUFACTURINGORDER' in filtered_261.columns else []

                # Use RAW 101 data for BFOUT (column name: BFOUT)
                if len(order_list) > 0:
                    filtered_101 = df_101_raw[df_101_raw['MANUFACTURINGORDER'].isin(order_list)]
                    total_hist_output_bf = filtered_101['BFOUT'].sum() if 'BFOUT' in filtered_101.columns else 0

            # Option 2: Use pre-computed data (deployment mode)
            elif has_precomputed_kd and kd_material in precomputed_data['kd_lookup']:
                kd_data = precomputed_data['kd_lookup'][kd_material]
                total_hist_output_bf = kd_data.get('total_bf_output', 0)
                # Estimate input from output using average yield (~90%)
                total_hist_input_bf = total_hist_output_bf / 0.9 if total_hist_output_bf > 0 else 0

            # Option 3: Fallback to joined data
            elif has_historical:
                filtered_hist = historical_data[historical_data['Input_Material'] == kd_material]
                if kd_thickness is not None and 'Input_Thickness' in filtered_hist.columns:
                    filtered_hist = filtered_hist[filtered_hist['Input_Thickness'] == kd_thickness]
                if kd_specie is not None and 'Input_Specie' in filtered_hist.columns:
                    filtered_hist = filtered_hist[filtered_hist['Input_Specie'] == kd_specie]
                if kd_grade is not None and 'Input_Grade' in filtered_hist.columns:
                    filtered_hist = filtered_hist[filtered_hist['Input_Grade'] == kd_grade]
                if kd_plant is not None and 'Input_Plant' in filtered_hist.columns:
                    filtered_hist = filtered_hist[filtered_hist['Input_Plant'] == kd_plant]
                unique_orders = filtered_hist.drop_duplicates('MANUFACTURINGORDER') if 'MANUFACTURINGORDER' in filtered_hist.columns else filtered_hist
                total_hist_input_bf = unique_orders['Total_Input_BF'].sum() if 'Total_Input_BF' in unique_orders.columns else 0
                total_hist_output_bf = filtered_hist['Total_Output_BF'].sum() if 'Total_Output_BF' in filtered_hist.columns else 0

            # Calculate output with wastage
            result = calculate_kd_output_with_wastage(
                input_bf=kd_input_bf,
                kd_distribution=kd_distribution,
                wastage_pct=kd_wastage
            )

            # Add historical totals to result
            result['total_hist_input_bf'] = total_hist_input_bf
            result['total_hist_output_bf'] = total_hist_output_bf

            # Store in session state
            st.session_state.kd_lookup_result = result
            st.session_state.kd_lookup_material = kd_material
            st.session_state.kd_lookup_run = True

    # Display Results
    if st.session_state.get('kd_lookup_run'):
        result = st.session_state.get('kd_lookup_result', {})
        input_material = st.session_state.get('kd_lookup_material', '')

        st.markdown("---")
        st.header("KD Output Materials")

        # Summary metrics
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            st.metric("Input Material", input_material)

        with col_s2:
            st.metric("Input Quantity", f"{result['input_bf']:,.0f} BF")

        with col_s3:
            st.metric("Wastage", f"{result['wastage_bf']:,.0f} BF ({result['wastage_pct']}%)")

        with col_s4:
            st.metric("Available for Output", f"{result['available_bf']:,.0f} BF")

        # Historical BF Totals (new row)
        st.markdown("**Historical Data Summary:**")
        col_h1, col_h2, col_h3 = st.columns(3)

        with col_h1:
            st.metric("Total BF Input (Historical)", f"{result.get('total_hist_input_bf', 0):,.0f}")

        with col_h2:
            st.metric("Total BF Output (Historical)", f"{result.get('total_hist_output_bf', 0):,.0f}")

        with col_h3:
            hist_yield = (result.get('total_hist_output_bf', 0) / result.get('total_hist_input_bf', 1)) * 100 if result.get('total_hist_input_bf', 0) > 0 else 0
            st.metric("Historical Yield", f"{hist_yield:.1f}%")

        st.markdown("---")

        # KD Materials Table
        st.subheader(f"Historical KD Outputs for {input_material}")

        kd_outputs = result.get('kd_outputs', [])

        if kd_outputs:
            # Sort by Expected BF descending (default)
            kd_outputs = sorted(kd_outputs, key=lambda x: x['Expected_Output_BF'], reverse=True)

            # Create DataFrame for display
            df_table = pd.DataFrame(kd_outputs)

            # Format the table for display
            df_display = pd.DataFrame({
                'KD Material': df_table['Output_Material'],
                'Grade': df_table['Output_Grade'],
                'BF Output': df_table['BF_Output'].apply(lambda x: f"{x:,.0f}"),
                'Historical %': df_table['Percentage'].apply(lambda x: f"{x:.1f}%"),
                'Avg Yield': df_table['Avg_Yield'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A"),
                'Expected BF': df_table['Expected_Output_BF'].apply(lambda x: f"{x:,.0f}")
            })

            # Display as a styled dataframe
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=min(400, 35 * len(df_display) + 38)
            )

            # Total row - show total historical BF output
            total_bf_output = sum(kd['BF_Output'] for kd in kd_outputs)
            st.markdown(f"**Total Historical BF Output: {total_bf_output:,.0f} | Available BF: {result['available_bf']:,.0f}**")

            st.markdown("---")

            # Selection for production order
            st.markdown("**Select KD materials for production order:**")

            # Add selection checkboxes via session state
            if 'selected_kds' not in st.session_state:
                st.session_state.selected_kds = []

            selected_kds = []
            # Create selection with multiselect
            material_options = [f"{kd['Output_Material']} | {kd['Output_Grade']} | {kd['Expected_Output_BF']:,.0f} BF" for kd in kd_outputs]
            selected_items = st.multiselect(
                "Select materials to include:",
                options=material_options,
                default=[],
                key="kd_multiselect"
            )

            # Map selections back to kd_outputs
            for i, option in enumerate(material_options):
                if option in selected_items:
                    selected_kds.append(kd_outputs[i])

            # Store selected KDs
            st.session_state.selected_kds = selected_kds

            # Production Order Section
            if selected_kds:
                st.markdown("---")
                st.subheader("Production Order Summary")

                # Calculate selected totals
                selected_bf = sum(kd['Expected_Output_BF'] for kd in selected_kds)
                selected_pct = sum(kd['Percentage'] for kd in selected_kds)

                # Display production order summary
                st.success(f"""
                **Production Order for {input_material}**

                - **Input Quantity:** {result['input_bf']:,.0f} BF
                - **Wastage ({result['wastage_pct']}%):** {result['wastage_bf']:,.0f} BF
                - **Available Output:** {result['available_bf']:,.0f} BF
                """)

                st.markdown("**Selected Outputs:**")

                order_data = []
                for kd in selected_kds:
                    order_data.append({
                        'KD Material': kd['Output_Material'],
                        'Grade': kd.get('Output_Grade', 'N/A'),
                        'Historical %': f"{kd['Percentage']:.1f}%",
                        'Avg Yield': f"{kd.get('Avg_Yield', 0):.1f}%",
                        'Planned Output (BF)': f"{kd['Expected_Output_BF']:,.0f}"
                    })

                order_df = pd.DataFrame(order_data)
                st.table(order_df)

                st.info(f"**Total Selected Output:** {selected_bf:,.0f} BF ({selected_pct:.1f}% of available)")

                # Export button
                export_data = {
                    'Input_Material': input_material,
                    'Input_Quantity_BF': result['input_bf'],
                    'Wastage_Pct': result['wastage_pct'],
                    'Wastage_BF': result['wastage_bf'],
                    'Available_Output_BF': result['available_bf'],
                    'Selected_Outputs': selected_kds
                }

                # Create CSV for download (includes Grade and Yield columns)
                export_rows = []
                for kd in selected_kds:
                    export_rows.append({
                        'Input_Material': input_material,
                        'Input_Quantity_BF': result['input_bf'],
                        'Wastage_Pct': result['wastage_pct'],
                        'Available_Output_BF': result['available_bf'],
                        'Output_Material': kd['Output_Material'],
                        'Output_Grade': kd.get('Output_Grade', 'N/A'),
                        'Historical_BF_Output': kd.get('BF_Output', 0),
                        'Historical_Order_Count': kd.get('Order_Count', 0),
                        'Historical_Percentage': kd['Percentage'],
                        'Avg_Yield_Pct': kd.get('Avg_Yield', 0),
                        'Yield_Std': kd.get('Yield_Std', 0),
                        'Planned_Output_BF': kd['Expected_Output_BF']
                    })

                export_df = pd.DataFrame(export_rows)
                csv = export_df.to_csv(index=False)

                st.download_button(
                    label="Download Production Order (CSV)",
                    data=csv,
                    file_name=f"production_order_{input_material}.csv",
                    mime="text/csv",
                    type="primary"
                )


def render_layman_explanation(summary: Dict, output_results: pd.DataFrame):
    """Render a simple, easy-to-understand explanation of the prediction."""
    st.subheader("What Does This Mean?")

    total_input = summary.get('total_input_bf', 0)
    best_yield = summary.get('best_yield_pct', 0)
    best_output = summary.get('best_output', 'N/A')
    best_output_bf = summary.get('best_output_bf', 0)
    avg_yield = summary.get('avg_yield_pct', 0)

    # Calculate waste
    waste_bf = total_input - best_output_bf
    waste_pct = 100 - best_yield

    # Simple explanation box
    st.info(f"""
    **In Simple Terms:**

    You are planning to use **{total_input:,.0f} board feet** of raw material.

    Based on historical production data and our prediction model:

    - **Best case scenario:** You can produce **{best_output_bf:,.0f} board feet** of finished product
      (specifically: {best_output})
    - **Expected yield:** For every 100 BF of raw material, you'll get approximately **{best_yield:.0f} BF** of finished product
    - **Material loss:** Approximately **{waste_bf:,.0f} BF** ({waste_pct:.1f}%) will be lost during production
      (sawdust, trim, defects, etc.)
    """)

    # Visual breakdown using columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Raw Material In")
        st.markdown(f"**{total_input:,.0f} BF**")
        st.caption("What you're putting into production")

    with col2:
        st.markdown("##### Production Process")
        st.markdown(f"**{best_yield:.1f}% Yield**")
        st.caption("Efficiency of conversion")

    with col3:
        st.markdown("##### Finished Product Out")
        st.markdown(f"**{best_output_bf:,.0f} BF**")
        st.caption("What you'll get as usable output")

    # Additional context
    with st.expander("How to interpret these results"):
        st.markdown(f"""
        **Understanding the Numbers:**

        1. **Yield Percentage ({best_yield:.1f}%)**: This is the efficiency rate.
           A yield of {best_yield:.1f}% means that for every 100 board feet of raw lumber you process,
           you'll end up with approximately {best_yield:.0f} board feet of finished product.

        2. **Material Loss ({waste_pct:.1f}%)**: The remaining {waste_pct:.1f}% is lost during production.
           This is normal and includes:
           - Sawdust and wood chips
           - Edge trimming
           - Defect removal (knots, splits, etc.)
           - Sizing adjustments

        3. **Output Material**: The finished product ({best_output}) is different from the raw material
           you started with. This is because lumber is transformed during manufacturing.

        **What affects yield?**
        - Raw material quality (grade, species)
        - Dimensions (thickness, length, width)
        - Target product specifications
        - Equipment and process efficiency

        **Using this for planning:**
        - If you need {best_output_bf:,.0f} BF of finished product, plan to have {total_input:,.0f} BF of raw material
        - Add 5-10% safety buffer for unexpected variations
        """)


def render_prediction_explanation(output_results: pd.DataFrame, input_materials: List[Dict], model_metadata: Dict):
    """
    Render a detailed explanation of how predictions were calculated.
    Shows the step-by-step process so users understand where numbers come from.
    """
    st.subheader("How Was This Prediction Calculated?")

    # Get info about the model
    trained_years = model_metadata.get('trained_years', [])
    best_model = model_metadata.get('best_model', 'Unknown')
    test_r2 = model_metadata.get('test_r2', 0)

    # Step 1: Your Inputs
    with st.expander("Step 1: Your Input Materials", expanded=True):
        st.markdown("**What you entered:**")

        for i, inp in enumerate(input_materials, 1):
            st.markdown(f"""
            **Input Material {i}:**
            - Material: `{inp.get('Input_Material', 'N/A')}`
            - Plant: `{inp.get('Input_Plant', 'N/A')}`
            - Thickness: `{inp.get('Input_Thickness', 'N/A')}`
            - Species: `{inp.get('Input_Specie', 'N/A')}`
            - Grade: `{inp.get('Input_Grade', 'N/A')}`
            - Quantity: **{inp.get('Total_Input_BF', 0):,.0f} BF**
            """)

        total_input = sum(m.get('Total_Input_BF', 0) for m in input_materials)
        st.info(f"**Total Input:** {total_input:,.0f} Board Feet")

    # Step 2: Historical Data Lookup
    with st.expander("Step 2: Historical Data Lookup", expanded=True):
        st.markdown(f"""
        **What the system did:**

        1. Searched the historical database (trained on **{', '.join(trained_years) if trained_years else 'available'}** data)
        2. Found all past manufacturing orders that used similar input materials
        3. For each past order, recorded:
           - What output material was produced
           - What was the yield percentage (output ÷ input)

        **Example from your data:**
        """)

        # Show a sample of the historical data used
        if len(output_results) > 0:
            sample = output_results.head(3)
            for _, row in sample.iterrows():
                hist_orders = row.get('Historical_Orders', 0)
                hist_yield = row.get('Historical_Yield_Pct', 0)
                output_mat = row.get('Output_Material', 'N/A')

                if hist_orders > 0:
                    st.markdown(f"""
                    - **{output_mat}**: Found **{hist_orders}** past orders with average yield of **{hist_yield:.1f}%**
                    """)

    # Step 3: ML Model Prediction
    with st.expander("Step 3: Machine Learning Prediction", expanded=True):
        st.markdown(f"""
        **How the ML model works:**

        1. **Model Type:** {best_model}
        2. **Model Accuracy:** R² = {test_r2:.2%} (higher is better, 100% would be perfect)
        3. **Training Data:** {', '.join(trained_years) if trained_years else 'Historical'} production records

        **What the model considers:**
        - Input material code (different materials have different yields)
        - Species (e.g., WO=White Oak, SM=Soft Maple have different characteristics)
        - Grade (1C, 2C, 3A - higher grades often have better yields)
        - Thickness (thicker lumber may yield differently)
        - Plant location (different equipment/processes)
        - Input quantity (larger batches may have different efficiency)

        **The model learned patterns like:**
        > "When input is Material X with Grade 2C and Thickness 6, the yield is typically around Y%"
        """)

    # Step 4: Combining Historical + ML
    with st.expander("Step 4: Final Prediction Calculation", expanded=True):
        st.markdown("""
        **How we combine the data:**

        The final predicted yield is a **weighted combination** of:

        1. **Historical Average** (what actually happened in the past)
        2. **ML Model Prediction** (what the model thinks will happen)

        **Formula:**
        ```
        Final Yield = (Weight × Historical Yield) + ((1 - Weight) × Model Yield)
        ```

        - If we have **lots of historical data** (50+ orders): Trust history more (70% historical, 30% model)
        - If we have **little historical data** (1-10 orders): Trust model more (30% historical, 70% model)

        **Then we calculate output:**
        ```
        Predicted Output (BF) = Input (BF) × (Predicted Yield % ÷ 100)
        ```
        """)

        # Show actual calculation for top result
        if len(output_results) > 0:
            top = output_results.iloc[0]
            pred_yield = top.get('Predicted_Yield_Pct', 0)
            hist_yield = top.get('Historical_Yield_Pct', 0)
            model_yield = top.get('Model_Yield_Pct', pred_yield)
            input_bf = top.get('Input_BF', total_input)
            output_bf = top.get('Predicted_Output_BF', 0)
            hist_orders = top.get('Historical_Orders', 0)

            st.markdown(f"""
            ---
            **Example Calculation (Top Result: {top.get('Output_Material', 'N/A')}):**

            - Historical Yield: **{hist_yield:.1f}%** (from {hist_orders} past orders)
            - ML Model Yield: **{model_yield:.1f}%**
            - Combined Yield: **{pred_yield:.1f}%**

            - Your Input: **{input_bf:,.0f} BF**
            - Predicted Output: {input_bf:,.0f} × {pred_yield:.1f}% = **{output_bf:,.0f} BF**
            """)

    # Step 5: Understanding Confidence
    with st.expander("Step 5: Why Some Predictions Are More Reliable", expanded=False):
        st.markdown("""
        **Confidence depends on:**

        | Factor | High Confidence | Low Confidence |
        |--------|-----------------|----------------|
        | Historical Orders | 50+ orders | 1-5 orders |
        | Yield Consistency | Low std deviation | High std deviation |
        | Model R² Score | > 0.7 | < 0.5 |

        **What to look for in the results table:**
        - **Historical Orders**: More orders = more reliable prediction
        - **Yield Std Dev**: Lower number = more consistent/predictable
        - **Confidence Level**: HIGH/MEDIUM/LOW indicator

        **Rule of thumb:**
        - ✅ **Trust predictions** with 20+ historical orders and low std dev
        - ⚠️ **Be cautious** with 1-5 historical orders (add safety margin)
        - ❌ **Very uncertain** if no historical data (model-only prediction)
        """)


def render_analysis_chart(output_results: pd.DataFrame):
    """Render horizontal bar chart of output materials vs yield."""
    if len(output_results) == 0:
        st.info("No output materials to display.")
        return

    # Show ALL output materials (no limit)
    display_df = output_results.copy()

    # Create color based on model type
    colors = ['#667eea' if t == 'Multi-Input' else '#38ef7d'
              for t in display_df['Model_Type']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=display_df['Output_Material'],
        x=display_df['Predicted_Yield_Pct'],
        orientation='h',
        marker_color=colors,
        text=[f"{y:.1f}%" for y in display_df['Predicted_Yield_Pct']],
        textposition='auto',
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Yield: %{x:.1f}%<br>" +
            "Output BF: %{customdata:,.0f}<br>" +
            "<extra></extra>"
        ),
        customdata=display_df['Predicted_Output_BF']
    ))

    fig.update_layout(
        title="Output Materials by Predicted Yield (%)",
        xaxis_title="Yield Percentage",
        yaxis_title="Output Material",
        height=max(400, len(display_df) * 25),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    **Color Legend:**
    - 🟪 Purple = Multi-Input Model
    - 🟩 Green = Single-Input Model
    """)


def render_detailed_results(output_results: pd.DataFrame):
    """Render detailed results table with ALL materials and full details."""
    if len(output_results) == 0:
        st.info("No results to display.")
        return

    st.markdown(f"**Showing all {len(output_results)} output materials**")

    # Define ALL columns we want to display (comprehensive list including actual output)
    display_cols = [
        'Input_Material',
        'Output_Material',
        'Input_BF',
        'Predicted_Output_BF',
        'Predicted_Yield_Pct',
        'Model_Yield_Pct',
        'Historical_Yield_Pct',
        'Yield_Std',
        'Historical_Orders',
        'Historical_Total_Output_BF',
        'Historical_Total_Input_BF',
        'Historical_Avg_Output_BF',
        'Model_Type'
    ]

    available_cols = [c for c in display_cols if c in output_results.columns]

    # Show ALL rows without any filtering
    st.dataframe(
        output_results[available_cols].rename(columns={
            'Input_Material': 'Input Materials Used',
            'Output_Material': 'Output Material',
            'Input_BF': 'Total Input (BF)',
            'Predicted_Output_BF': 'Predicted Output (BF)',
            'Predicted_Yield_Pct': 'Predicted Yield %',
            'Model_Yield_Pct': 'Model Yield %',
            'Historical_Yield_Pct': 'Historical Yield %',
            'Yield_Std': 'Yield Std Dev',
            'Historical_Orders': 'Historical Orders',
            'Historical_Total_Output_BF': 'Hist. Total Output (BF)',
            'Historical_Total_Input_BF': 'Hist. Total Input (BF)',
            'Historical_Avg_Output_BF': 'Hist. Avg Output (BF)',
            'Model_Type': 'Model Type'
        }),
        use_container_width=True,
        hide_index=True,
        height=min(800, max(400, len(output_results) * 35))  # Dynamic height based on row count
    )

    # Download button
    csv = output_results.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="yield_predictions.csv",
        mime="text/csv"
    )


def render_predicted_outputs(predicted_outputs: List[Dict]):
    """
    Render ML-predicted output materials with probabilities.

    This shows the classification model's predictions for which output
    materials are most likely to be produced from the given inputs.
    """
    if not predicted_outputs:
        st.info("No output material predictions available. Train the classifier to enable this feature.")
        return

    st.subheader("ML Predicted Output Materials")
    st.caption("Classification model predictions - shows most likely outputs based on your input characteristics")

    # Create a visual display of predictions
    for pred in predicted_outputs:
        col1, col2, col3 = st.columns([4, 3, 1])

        with col1:
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][pred['rank'] - 1] if pred['rank'] <= 5 else f"{pred['rank']}."
            st.write(f"{rank_emoji} **{pred['output_material']}**")

        with col2:
            # Progress bar showing probability
            st.progress(min(pred['probability'], 1.0))

        with col3:
            st.write(f"**{pred['probability_pct']:.1f}%**")

    # Add explanation
    with st.expander("How does output prediction work?"):
        st.markdown("""
        **What is this?**

        This section uses a **classification model** to predict which output materials
        are most likely to be produced based on your input material characteristics.

        **How it works:**
        - The model learned from historical production data
        - It identifies patterns between input materials and their outputs
        - Predictions are ranked by probability/confidence

        **How to use this:**
        - Higher probability = more likely output based on historical patterns
        - Use this to plan which finished products to expect
        - Cross-reference with the Historical Analysis below for yield estimates
        """)


def render_recommendation(recommendation: Dict, total_input_bf: float):
    """
    Render the top recommendation with confidence and risk indicators.

    This is the main decision-support display showing:
    - Recommended output material
    - Expected yield with confidence range
    - Risk level assessment
    """
    if not recommendation.get('has_recommendation'):
        st.warning(recommendation.get('message', 'No recommendation available.'))
        return

    st.subheader("Recommendation")

    # Determine color based on strength
    strength = recommendation.get('strength', 'WEAK')
    if strength == 'STRONG':
        box_color = "#28a745"  # Green
    elif strength == 'MODERATE':
        box_color = "#ffc107"  # Yellow
    else:
        box_color = "#dc3545"  # Red

    # Main recommendation box
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {box_color}22 0%, {box_color}11 100%);
        border-left: 4px solid {box_color};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    ">
        <h3 style="margin: 0; color: {box_color};">
            {recommendation['emoji']} Produce: {recommendation['output_material']}
        </h3>
        <p style="font-size: 1.1em; margin: 10px 0;">
            <strong>Expected Output:</strong> {recommendation['expected_output_bf']:,.0f} BF
            <span style="color: #666;">({recommendation['output_range_bf']})</span>
        </p>
        <p style="font-size: 1.1em; margin: 10px 0;">
            <strong>Expected Yield:</strong> {recommendation['expected_yield']:.1f}%
            <span style="color: #666;">({recommendation['yield_range']})</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence & Risk metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_color = {
            'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🟠', 'VERY LOW': '🔴'
        }.get(recommendation['confidence_level'], '⚪')
        st.metric(
            label="Confidence Level",
            value=f"{confidence_color} {recommendation['confidence_level']}",
            help="Based on historical data quantity, yield consistency, and model performance"
        )

    with col2:
        risk_color = {
            'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'VERY HIGH': '🔴'
        }.get(recommendation['risk_level'], '⚪')
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {recommendation['risk_level']}",
            help="Lower risk = more predictable results"
        )

    with col3:
        st.metric(
            label="Confidence Score",
            value=f"{recommendation['confidence_score']}/100",
            help="Combined score from data, consistency, and model factors"
        )

    with col4:
        st.metric(
            label="Historical Orders",
            value=f"{recommendation['historical_orders']:,}",
            help="Number of similar production orders in historical data"
        )

    # Explanation
    with st.expander("Why this recommendation?"):
        # Calculate production loss
        production_loss_pct = 100 - recommendation['expected_yield']
        total_input_bf = recommendation['expected_output_bf'] / (recommendation['expected_yield'] / 100) if recommendation['expected_yield'] > 0 else 0
        loss_bf = total_input_bf - recommendation['expected_output_bf']

        # Get average yield for comparison (approximately 91-92% based on data)
        avg_yield = 91.5  # Typical average from training data
        avg_output = total_input_bf * (avg_yield / 100)

        # Safely get yield_std and model_r2 (handle missing or non-numeric values)
        yield_std_val = recommendation.get('yield_std', 0)
        if not isinstance(yield_std_val, (int, float)):
            yield_std_val = 0
        yield_std_display = f"{yield_std_val:.1f}" if yield_std_val > 0 else "N/A"

        model_r2_val = recommendation.get('model_r2', 0.3)
        if not isinstance(model_r2_val, (int, float)):
            model_r2_val = 0.3

        # Determine impact indicators
        yield_std_impact = '🟢 Consistent' if yield_std_val <= 10 else '🟠 Variable' if yield_std_val <= 20 else '🔴 Highly Variable'
        model_r2_impact = '🟢 Good' if model_r2_val >= 0.5 else '🟠 Moderate' if model_r2_val >= 0.3 else '🔴 Limited'

        st.markdown(f"""
        ### Step 1: What We Found in Historical Data

        We searched past production records and found **{recommendation['historical_orders']} order(s)**
        where similar input materials were used to produce **{recommendation['output_material']}**.
        """)

        # Warning for low historical data
        if recommendation['historical_orders'] <= 5:
            st.warning(f"""
            **⚠️ Limited Data Warning**

            Only **{recommendation['historical_orders']}** historical order(s) found. This means:
            - The prediction is based on very few data points
            - The actual yield could be significantly different
            - Consider using the **average yield ({avg_yield:.1f}%)** as a safer estimate

            **Safer Estimate:** {avg_output:,.0f} BF output (using {avg_yield:.1f}% average yield)
            """)

        st.markdown(f"""
        ### Step 2: How Yield is Calculated

        **Yield** = (Output BF ÷ Input BF) × 100

        | Metric | Value |
        |--------|-------|
        | Your Input | {total_input_bf:,.0f} BF |
        | Predicted Output | {recommendation['expected_output_bf']:,.0f} BF |
        | **Predicted Yield** | **{recommendation['expected_yield']:.1f}%** |

        ### Step 3: Production Loss Explained

        In any production process, some material is lost due to:
        - Sawdust and cutting waste
        - Trimming and edge losses
        - Defects and rejects
        - Moisture content changes

        | Loss Calculation | Value |
        |-----------------|-------|
        | Input Material | {total_input_bf:,.0f} BF |
        | Expected Output | {recommendation['expected_output_bf']:,.0f} BF |
        | **Expected Loss** | **{loss_bf:,.0f} BF ({production_loss_pct:.1f}%)** |

        ### Step 4: Why This Confidence Level?

        **Confidence Score: {recommendation['confidence_score']}/100** = **{recommendation['confidence_level']}**

        The score is based on 3 factors:

        | Factor | Your Data | Impact |
        |--------|-----------|--------|
        | Historical Orders | {recommendation['historical_orders']} | {'🟢 Good (20+)' if recommendation['historical_orders'] >= 20 else '🟠 Limited (5-19)' if recommendation['historical_orders'] >= 5 else '🔴 Very Limited (<5)'} |
        | Yield Consistency | {yield_std_display}% std dev | {yield_std_impact} |
        | Model R² | {model_r2_val:.2f} | {model_r2_impact} |

        ### Step 5: Planning Recommendation

        | Scenario | Output (BF) | Yield |
        |----------|-------------|-------|
        | **Best Estimate** | {recommendation['expected_output_bf']:,.0f} | {recommendation['expected_yield']:.1f}% |
        | Optimistic | {recommendation['output_range_bf'].split(' - ')[1]} | {recommendation['yield_range'].split(' - ')[1]} |
        | Conservative | {recommendation['output_range_bf'].split(' - ')[0]} | {recommendation['yield_range'].split(' - ')[0]} |
        """)

        # Additional recommendation for low confidence
        if recommendation['confidence_level'] in ['LOW', 'VERY LOW']:
            st.info(f"""
            **💡 Recommendation for Low Confidence Predictions**

            Since confidence is {recommendation['confidence_level']}, consider:
            1. **Use conservative estimate**: Plan for {recommendation['output_range_bf'].split(' - ')[0]} output
            2. **Use average yield**: {avg_yield:.1f}% → {avg_output:,.0f} BF output
            3. **Build in buffer**: Order extra input material to account for uncertainty
            """)


def render_enhanced_results(output_results: pd.DataFrame):
    """
    Render enhanced detailed results with confidence and risk columns.
    """
    if len(output_results) == 0:
        st.info("No results to display.")
        return

    st.markdown(f"**All {len(output_results)} Output Options (sorted by recommendation score)**")

    # Define columns for enhanced display
    display_cols = [
        'Output_Material',
        'Predicted_Yield_Pct',
        'Yield_Min_Pct',
        'Yield_Max_Pct',
        'Predicted_Output_BF',
        'Confidence_Level',
        'Risk_Level',
        'Confidence_Score',
        'Historical_Orders',
        'Historical_Yield_Pct',
        'Yield_Std',
        'Recommendation_Score'
    ]

    available_cols = [c for c in display_cols if c in output_results.columns]

    # If enhanced columns not available, fall back to basic display
    if 'Confidence_Level' not in output_results.columns:
        render_detailed_results(output_results)
        return

    # Rename for display
    display_df = output_results[available_cols].rename(columns={
        'Output_Material': 'Output Material',
        'Predicted_Yield_Pct': 'Yield %',
        'Yield_Min_Pct': 'Min Yield %',
        'Yield_Max_Pct': 'Max Yield %',
        'Predicted_Output_BF': 'Output (BF)',
        'Confidence_Level': 'Confidence',
        'Risk_Level': 'Risk',
        'Confidence_Score': 'Score',
        'Historical_Orders': 'Hist. Orders',
        'Historical_Yield_Pct': 'Hist. Yield %',
        'Yield_Std': 'Std Dev',
        'Recommendation_Score': 'Rec. Score'
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(600, max(300, len(output_results) * 35))
    )

    # Color legend
    st.markdown("""
    **Confidence Levels:**
    🟢 HIGH (80+) = Very reliable prediction |
    🟡 MEDIUM (60-79) = Reasonable confidence |
    🟠 LOW (40-59) = Some uncertainty |
    🔴 VERY LOW (<40) = High uncertainty
    """)

    # Download
    csv = output_results.to_csv(index=False)
    st.download_button(
        label="Download Full Results (CSV)",
        data=csv,
        file_name="yield_predictions_enhanced.csv",
        mime="text/csv"
    )


def render_model_performance(metrics: Dict):
    """Render model performance metrics."""
    if not metrics:
        st.info("No model metrics available.")
        return

    st.subheader("Model Performance Comparison")

    rows = []
    for name, m in metrics.items():
        if isinstance(m, dict):
            rows.append({
                'Model': name,
                'Train R²': round(m.get('train_r2', 0), 4),
                'Test R²': round(m.get('test_r2', m.get('R2', 0)), 4),
                'MAE': round(m.get('MAE', 0), 2),
                'RMSE': round(m.get('RMSE', 0), 2),
                'MAPE': f"{m.get('MAPE', 0):.2f}%" if m.get('MAPE') else 'N/A'
            })

    if rows:
        perf_df = pd.DataFrame(rows).sort_values('Test R²', ascending=False)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Highlight best model
        best = perf_df.iloc[0]
        st.success(f"**Best Model:** {best['Model']} with Test R² = {best['Test R²']}")


def render_test_results(test_results: pd.DataFrame):
    """Render actual vs predicted comparison."""
    if test_results is None or len(test_results) == 0:
        st.info("No test results available.")
        return

    st.subheader("Actual vs Predicted (Test Set)")

    # Sample for display
    if len(test_results) > 500:
        display_df = test_results.sample(500, random_state=42)
    else:
        display_df = test_results

    # Scatter plot
    fig = px.scatter(
        display_df,
        x='Actual',
        y='Predicted',
        title='Actual vs Predicted Yield',
        labels={'Actual': 'Actual Yield %', 'Predicted': 'Predicted Yield %'},
        opacity=0.5
    )

    # Add perfect prediction line
    max_val = max(display_df['Actual'].max(), display_df['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    if 'Error' in display_df.columns:
        fig_hist = px.histogram(
            display_df,
            x='Error',
            nbins=50,
            title='Prediction Error Distribution'
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Clear cached data to pick up code changes (especially for raw CSV loading)
    # This ensures df_261_raw and df_101_raw are properly loaded
    if 'cache_cleared' not in st.session_state:
        st.cache_data.clear()
        st.session_state.cache_cleared = True

    st.markdown('<h1 class="main-header">Production Planning - Material Yield Prediction</h1>',
                unsafe_allow_html=True)

    # Get available pre-trained model years
    available_model_years = get_available_model_years()

    # Auto-select model (sidebar model selection hidden)
    if available_model_years:
        selected_model_year = '2025' if '2025' in available_model_years else available_model_years[-1]
    else:
        selected_model_year = None

    # Load artifacts for selected model year
    artifacts = load_model_artifacts(selected_model_year)
    # Load historical data for the selected year (also returns raw 261/101 data for KD lookup)
    historical_data, df_261_raw, df_101_raw, precomputed_data = load_historical_data(selected_model_year)

    # Sidebar model info hidden — using auto-selected model

    # Check if model is loaded
    if artifacts is None:
        st.error("""
        **Model not found!**

        Please train the model first by clicking "Train Model" in the sidebar,
        or ensure your CSV files (101.csv and 261.csv) are in the current directory.
        """)
        return

    model = artifacts['model']
    encoders = artifacts['encoders']
    feature_columns = artifacts['feature_columns']
    metrics = artifacts['metrics']
    test_results = artifacts['test_results']
    classifier = artifacts.get('classifier')
    classifier_features = artifacts.get('classifier_features', [])
    classifier_source = artifacts.get('classifier_source', 'missing')

    # Load dropdown options for reverse prediction
    options = load_dropdown_options()

    # Prediction Mode Selection
    st.header("1. Select Prediction Mode")
    prediction_mode = st.radio(
        "Choose prediction type:",
        [
            "V2.1",
            "V2.2",
            "Advanced Forward Prediction (Deprecated)",
            "Multi-Output Prediction (Deprecated)"
        ],
        horizontal=True,
        help="V2.1: ML-only yield + ML output materials. V2.2: External API prediction with customizable options. Advanced (Deprecated): ML + statistical KD distribution. Multi-Output (Deprecated): Upload CSV for board-level output distribution."
    )

    st.markdown("---")

    if prediction_mode == "Sample Data Test":
        # Sample Data Test Section
        render_sample_data_test_section(model, encoders, feature_columns)
        return  # Exit early for sample data test mode

    if prediction_mode == "KD Material Lookup (Find KD Outputs)":
        # KD Material Lookup Section
        render_kd_material_lookup_section(options, historical_data, df_261_raw, df_101_raw, precomputed_data)
        return  # Exit early for KD lookup mode

    if prediction_mode == "Material Level Forward Prediction":
        # Material Level Forward Prediction Section (Yield Recommendation Engine)
        render_material_level_forward_prediction_section(options, df_261_raw, df_101_raw, precomputed_data)
        return  # Exit early for this mode

    if prediction_mode == "Advanced Forward Prediction (Deprecated)":
        # Advanced Forward Prediction Section (ML + Statistical Hybrid)
        render_advanced_forward_prediction_section(options, df_261_raw, df_101_raw, model, encoders, feature_columns, precomputed_data)
        return  # Exit early for this mode

    if prediction_mode == "V2.1":
        # V2.1 Section (ML-only KD material distribution)
        kd_artifacts = load_kd_distribution_model()
        render_test2_ml_only_section(
            options=options,
            model=model,
            classifier=classifier,
            encoders=encoders,
            feature_columns=feature_columns,
            classifier_features=classifier_features,
            classifier_source=classifier_source,
            df_261_raw=df_261_raw,
            df_101_raw=df_101_raw,
            precomputed_data=precomputed_data,
            kd_model_artifacts=kd_artifacts,
        )
        return  # Exit early for this mode

    if prediction_mode == "Multi-Output Prediction (Deprecated)":
        # Multi-Output Prediction Section (CSV Upload + Board Distribution)
        render_multi_output_prediction_section()
        return  # Exit early for this mode

    if prediction_mode == "V2.2":
        # V2.2 Section (External API prediction)
        render_final_v2_section()
        return  # Exit early for this mode

    if prediction_mode == "Reverse Prediction (Output -> Input)":
        # Reverse Prediction Section
        render_reverse_prediction_section(model, encoders, feature_columns, options)
        return  # Exit early for reverse prediction mode

    # Forward Prediction Section (original flow)
    st.header("2. Enter Input Materials")
    input_materials = render_input_section()

    # Run Prediction Button
    st.markdown("---")
    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not input_materials or all(m.get('Total_Input_BF', 0) == 0 for m in input_materials):
            st.error("Please enter at least one input material with quantity > 0")
        else:
            with st.spinner("Running prediction..."):
                try:
                    # Calculate total input
                    total_input_bf = sum(m.get('Total_Input_BF', 0) for m in input_materials)

                    # Use ENHANCED simulation with confidence levels
                    if historical_data is not None and len(historical_data) > 0:
                        output_results = simulate_output_materials_enhanced(
                            model,
                            input_materials,
                            encoders,
                            feature_columns,
                            historical_data,
                            model_metrics=metrics  # Pass metrics for confidence calculation
                        )
                    else:
                        # Fallback: simple prediction
                        output_results = pd.DataFrame()
                        for inp in input_materials:
                            pred = forward_predict(model, inp, encoders, feature_columns)
                            output_results = pd.concat([output_results, pd.DataFrame([{
                                'Output_Material': 'Predicted Output',
                                'Predicted_Yield_Pct': pred['predicted_yield_pct'],
                                'Predicted_Output_BF': pred['predicted_output_bf'],
                                'Historical_Yield_Pct': pred['predicted_yield_pct'],
                                'Historical_Orders': 0,
                                'Model_Type': 'Single-Input',
                                'Confidence_Level': 'LOW',
                                'Risk_Level': 'HIGH',
                                'Confidence_Score': 30
                            }])], ignore_index=True)

                    # Get summary
                    summary = get_prediction_summary(output_results, total_input_bf, metrics)

                    # Get top recommendation
                    recommendation = get_top_recommendation(output_results, total_input_bf)

                    # Run Output Material Classification (if classifier available)
                    predicted_outputs = []
                    classifier = artifacts.get('classifier')
                    classifier_features = artifacts.get('classifier_features', [])

                    if classifier is not None and len(input_materials) > 0:
                        try:
                            predicted_outputs = predict_output_material(
                                classifier,
                                input_materials[0],  # Use first input material for classification
                                encoders,
                                classifier_features,
                                top_k=5
                            )
                        except Exception as clf_error:
                            print(f"Classifier prediction error: {clf_error}")

                    # Store in session state
                    st.session_state.output_results = output_results
                    st.session_state.prediction_summary = summary
                    st.session_state.predicted_outputs = predicted_outputs
                    st.session_state.recommendation = recommendation
                    st.session_state.total_input_bf = total_input_bf
                    st.session_state.input_materials_used = input_materials  # Store for explanation
                    st.session_state.prediction_run = True

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display Results
    if st.session_state.get('prediction_run'):
        st.markdown("---")
        st.header("2. Prediction Results")

        output_results = st.session_state.get('output_results', pd.DataFrame())
        summary = st.session_state.get('prediction_summary', {})
        predicted_outputs = st.session_state.get('predicted_outputs', [])
        recommendation = st.session_state.get('recommendation', {})
        total_input_bf = st.session_state.get('total_input_bf', 0)

        # TOP RECOMMENDATION (Most Important - Show First!)
        render_recommendation(recommendation, total_input_bf)

        st.markdown("---")

        # Summary metrics
        render_prediction_summary(summary, output_results)

        # Layman explanation (easy to understand section)
        render_layman_explanation(summary, output_results)

        # ML Predicted Outputs (if classifier available)
        if predicted_outputs:
            st.markdown("---")
            render_predicted_outputs(predicted_outputs)

        st.markdown("---")
        st.subheader("All Output Options & Analysis")

        # Tabs for detailed views - use enhanced results
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Enhanced Results",
            "How It Was Calculated",
            "Analysis Chart",
            "Basic Results",
            "Model Performance",
            "Test Set Results"
        ])

        with tab1:
            render_enhanced_results(output_results)

        with tab2:
            # Get model metadata and input materials for explanation
            model_metadata = artifacts.get('model_metadata', {})
            input_materials_used = st.session_state.get('input_materials_used', [])
            render_prediction_explanation(output_results, input_materials_used, model_metadata)

        with tab3:
            render_analysis_chart(output_results)

        with tab4:
            render_detailed_results(output_results)

        with tab5:
            render_model_performance(metrics)

        with tab6:
            render_test_results(test_results)


if __name__ == "__main__":
    main()
