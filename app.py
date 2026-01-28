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

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules from src/
from src.data_preparation import (
    prepare_full_dataset,
    prepare_full_dataset_with_raw,
    load_raw_csv_data,
    get_unique_materials_from_csv,
    get_historical_yield_by_material,
    get_available_years
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
    get_advanced_forward_prediction
)

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
    """Get list of years with trained models."""
    years = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.isdigit():
                # Check if model exists in this folder
                if os.path.exists(os.path.join(item_path, "yield_model.joblib")):
                    years.append(item)
    return sorted(years)


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

        classifier_path = os.path.join(base_path, "output_classifier.joblib")
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)

        clf_features_path = os.path.join(base_path, "classifier_features.joblib")
        if os.path.exists(clf_features_path):
            classifier_features = joblib.load(clf_features_path)

        clf_metrics_path = os.path.join(base_path, "classifier_metrics.joblib")
        if os.path.exists(clf_metrics_path):
            classifier_metrics = joblib.load(clf_metrics_path)

        return {
            'model': model,
            'encoders': encoders,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'test_results': test_results,
            'classifier': classifier,
            'classifier_features': classifier_features,
            'classifier_metrics': classifier_metrics,
            'model_metadata': model_metadata
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_historical_data(selected_year: str = None):
    """Load and prepare historical data for the selected year.

    Args:
        selected_year: Year to load data for (e.g., '2024', '2025').
                      If None, loads all available years.

    Returns:
        Tuple of (historical_data, df_261_raw, df_101_raw)
        - historical_data: Joined and encoded data (for distribution calculation)
        - df_261_raw: RAW 261 CSV data (for accurate BFIN totals)
        - df_101_raw: RAW 101 CSV data (for accurate BFOUT totals)
    """
    # Determine which years to load
    if selected_year:
        years_to_load = [selected_year]
    else:
        years_to_load = get_available_years()

    # Load RAW CSV data for accurate totals (no cleaning/aggregation)
    df_261_raw, df_101_raw = None, None
    try:
        df_261_raw, df_101_raw = load_raw_csv_data(years=years_to_load)
    except Exception as e:
        print(f"Could not load raw CSV data: {e}")

    # Option 1: Load from year-specific CSV files
    try:
        df, encoders = prepare_full_dataset(years=years_to_load)
        print(f"Loaded historical data for {years_to_load}: {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        return df, df_261_raw, df_101_raw
    except Exception as e:
        print(f"Could not load year-specific data: {e}")

    # Option 2: Try loading from generic CSV files (101.csv, 261.csv)
    try:
        csv_101 = os.path.join(DATA_DIR, "101.csv")
        csv_261 = os.path.join(DATA_DIR, "261.csv")
        if os.path.exists(csv_101) and os.path.exists(csv_261):
            df, encoders = prepare_full_dataset(csv_261, csv_101)
            return df, df_261_raw, df_101_raw
    except Exception as e:
        print(f"Could not load from generic CSV: {e}")

    # Option 3: Load pre-computed historical summary from models/ folder (no raw data)
    try:
        hist_path = os.path.join(MODELS_DIR, "historical_summary.joblib")
        if os.path.exists(hist_path):
            historical_summary = joblib.load(hist_path)
            historical_summary = historical_summary.rename(columns={
                'Mean_Yield': 'Yield_Percentage',
                'Order_Count': 'Historical_Orders'
            })
            print(f"Loaded historical_summary.joblib: {len(historical_summary)} records")
            return historical_summary, df_261_raw, df_101_raw
    except Exception as e:
        print(f"Could not load historical_summary: {e}")

    return None, df_261_raw, df_101_raw


@st.cache_data
def load_dropdown_options():
    """Load options for dropdown menus from CSV files."""
    options = get_unique_materials_from_csv()
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


def render_material_level_forward_prediction_section(options, df_261_raw, df_101_raw):
    """Render the Material Level Forward Prediction (Yield Recommendation Engine).

    This mode allows users to:
    - Enter KS material, plant, and input BF quantity
    - Get predicted total output BF based on historical yield
    - Get distribution across KD materials (summing to 100%)
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
        if df_261_raw is None or df_101_raw is None:
            st.error("No raw data available. Please ensure data files are loaded.")
            return

        if selected_material == 'No materials found':
            st.error("Please select a valid KS material.")
            return

        with st.spinner("Calculating prediction..."):
            result = get_material_level_forward_prediction(
                ks_material=selected_material,
                plant=selected_plant,
                input_bf=input_bf,
                df_261_raw=df_261_raw,
                df_101_raw=df_101_raw,
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


def render_advanced_forward_prediction_section(options, df_261_raw, df_101_raw, model, encoders, feature_columns):
    """Render the Advanced Forward Prediction (ML + Statistical Hybrid).

    Uses ML model for yield prediction + statistical distribution for KD materials.
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
        if df_261_raw is None or df_101_raw is None:
            st.error("No raw data available. Please ensure data files are loaded.")
            return

        if selected_material == 'No materials found':
            st.error("Please select a valid KS material.")
            return

        if model is None:
            st.error("ML model not available. Please ensure the model is trained.")
            return

        with st.spinner("Running ML prediction..."):
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


def render_kd_material_lookup_section(options, historical_data, df_261_raw=None, df_101_raw=None):
    """Render the KD Material Lookup section - find historical KD outputs for a KS input.

    Args:
        options: Dropdown options dict
        historical_data: Joined historical data (for distribution calculation)
        df_261_raw: RAW 261 CSV data (for accurate BFIN totals matching notebook)
        df_101_raw: RAW 101 CSV data (for accurate BFOUT totals matching notebook)
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
        if historical_data is None or len(historical_data) == 0:
            st.error("No historical data available. Please ensure data files are loaded.")
            return

        if kd_material == 'No materials found':
            st.error("Please select a valid input material.")
            return

        with st.spinner("Looking up historical KD outputs..."):
            # Get historical distribution
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

            # Calculate total historical BF Input/Output using RAW CSV data
            # This matches the notebook approach - no cleaning/aggregation
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
                else:
                    total_hist_output_bf = 0
            else:
                # Fallback to joined data if raw data not available
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

    # Sidebar
    with st.sidebar:
        # =====================================================================
        # MODEL SELECTION SECTION
        # =====================================================================
        if available_model_years:
            st.header("Model Selection")
            selected_model_year = st.selectbox(
                "Select prediction model:",
                options=available_model_years,
                index=len(available_model_years) - 1,  # Default to latest year
                help="Choose which year's trained model to use for predictions"
            )
            st.markdown("---")
        else:
            selected_model_year = None

    # Load artifacts for selected model year
    artifacts = load_model_artifacts(selected_model_year)
    # Load historical data for the selected year (also returns raw 261/101 data for KD lookup)
    historical_data, df_261_raw, df_101_raw = load_historical_data(selected_model_year)

    # Continue sidebar content
    with st.sidebar:
        # =====================================================================
        # CURRENT MODEL INFO SECTION
        # =====================================================================
        st.header("Current Model")

        if artifacts:
            # Show which model year is selected
            if selected_model_year:
                st.success(f"Using {selected_model_year} Model")
            else:
                st.success("Model Loaded")

            # Show training years from metadata
            model_metadata = artifacts.get('model_metadata', {})
            trained_years = model_metadata.get('trained_years', [])
            if trained_years:
                st.info(f"Trained on: {', '.join(trained_years)}")

            # Show trained timestamp
            trained_at = model_metadata.get('trained_at', '')
            if trained_at:
                # Format the timestamp nicely
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(trained_at)
                    st.caption(f"Trained: {dt.strftime('%Y-%m-%d %H:%M')}")
                except:
                    pass

            # Show model metrics
            if artifacts['metrics']:
                best_r2 = max(
                    m.get('test_r2', m.get('R2', 0))
                    for m in artifacts['metrics'].values()
                    if isinstance(m, dict)
                )
                st.metric("Model R² Score", f"{best_r2:.4f}")

            # Show classifier info if available
            if artifacts.get('classifier') is not None:
                st.success("Output Classifier Loaded")
                classifier_metrics = artifacts.get('classifier_metrics', {})
                if classifier_metrics:
                    best_acc = max(
                        m.get('accuracy', 0)
                        for m in classifier_metrics.values()
                        if isinstance(m, dict)
                    )
                    st.metric("Classifier Accuracy", f"{best_acc:.2%}")
            else:
                st.warning("Output Classifier not trained")
        else:
            st.warning("No model found for selected year")
            st.caption("Please select a different year or ensure model files exist")

        st.markdown("---")
        st.markdown("""
        **SAP Logic:**
        - 261 = Goods Issue = Input (raw materials consumed)
        - 101 = Goods Receipt = Output (finished goods produced)
        - Yield = Output (101) / Input (261)
        """)

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

    # Load dropdown options for reverse prediction
    options = load_dropdown_options()

    # Prediction Mode Selection
    st.header("1. Select Prediction Mode")
    prediction_mode = st.radio(
        "Choose prediction type:",
        [
            "KD Material Lookup (Find KD Outputs)",
            "Material Level Forward Prediction",
            "Advanced Forward Prediction",
            "Forward Prediction (Input -> Output)",
            "Reverse Prediction (Output -> Input)"
        ],
        horizontal=True,
        help="KD Lookup: Find historical KD outputs. Material Level: Statistical prediction from historical data. Advanced: ML model for yield + statistical distribution. Forward: Full ML prediction. Reverse: Calculate required input."
    )

    st.markdown("---")

    if prediction_mode == "KD Material Lookup (Find KD Outputs)":
        # KD Material Lookup Section
        render_kd_material_lookup_section(options, historical_data, df_261_raw, df_101_raw)
        return  # Exit early for KD lookup mode

    if prediction_mode == "Material Level Forward Prediction":
        # Material Level Forward Prediction Section (Yield Recommendation Engine)
        render_material_level_forward_prediction_section(options, df_261_raw, df_101_raw)
        return  # Exit early for this mode

    if prediction_mode == "Advanced Forward Prediction":
        # Advanced Forward Prediction Section (ML + Statistical Hybrid)
        render_advanced_forward_prediction_section(options, df_261_raw, df_101_raw, model, encoders, feature_columns)
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
