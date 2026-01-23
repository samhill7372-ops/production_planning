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
    calculate_confidence_level
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


@st.cache_resource
def load_model_artifacts():
    """Load trained model and artifacts from models/ folder."""
    import json

    try:
        model_path = os.path.join(MODELS_DIR, "yield_model.joblib")
        if not os.path.exists(model_path):
            return None

        model = joblib.load(model_path)
        encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "encoders.joblib")) else {}
        feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "feature_columns.joblib")) else []
        metrics = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "metrics.joblib")) else {}
        test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv")) if os.path.exists(os.path.join(DATA_DIR, "test_results.csv")) else None

        # Load model metadata (includes training years)
        model_metadata = {}
        metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        # Load Output Material Classifier (if available)
        classifier = None
        classifier_features = []
        classifier_metrics = {}

        classifier_path = os.path.join(MODELS_DIR, "output_classifier.joblib")
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)

        clf_features_path = os.path.join(MODELS_DIR, "classifier_features.joblib")
        if os.path.exists(clf_features_path):
            classifier_features = joblib.load(clf_features_path)

        clf_metrics_path = os.path.join(MODELS_DIR, "classifier_metrics.joblib")
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
def load_historical_data():
    """Load and prepare historical data."""
    # Option 1: Load from CSV files in data/ folder (local development)
    try:
        csv_101 = os.path.join(DATA_DIR, "101.csv")
        csv_261 = os.path.join(DATA_DIR, "261.csv")
        if os.path.exists(csv_101) and os.path.exists(csv_261):
            df, encoders = prepare_full_dataset(csv_261, csv_101)
            return df
    except Exception as e:
        print(f"Could not load from CSV: {e}")

    # Option 2: Load pre-computed historical summary from models/ folder
    try:
        hist_path = os.path.join(MODELS_DIR, "historical_summary.joblib")
        if os.path.exists(hist_path):
            historical_summary = joblib.load(hist_path)
            # Rename columns to match expected format
            historical_summary = historical_summary.rename(columns={
                'Mean_Yield': 'Yield_Percentage',
                'Order_Count': 'Historical_Orders'
            })
            print(f"Loaded historical_summary.joblib: {len(historical_summary)} input-output combinations")
            return historical_summary
    except Exception as e:
        print(f"Could not load historical_summary: {e}")

    return None


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
    st.markdown('<h1 class="main-header">Production Planning - Material Yield Prediction</h1>',
                unsafe_allow_html=True)

    # Load artifacts
    artifacts = load_model_artifacts()
    # Load historical data (uses caching, will work locally with CSVs)
    historical_data = load_historical_data()

    # Sidebar
    with st.sidebar:
        # =====================================================================
        # DATA UPLOAD SECTION
        # =====================================================================
        st.header("Data Upload")

        with st.expander("Upload CSV Files", expanded=False):
            st.markdown("""
            Upload your SAP data files to train the model.

            **Required for each year:**
            - **261 file**: Input materials (Goods Issue)
            - **101 file**: Output materials (Goods Receipt)

            *Both files must be uploaded for a year to be available for training.*
            """)

            # Year input
            upload_year = st.text_input(
                "Data Year",
                value="2025",
                help="Year for the uploaded data (e.g., 2024, 2025)"
            )

            # File uploaders
            file_261 = st.file_uploader(
                "261 (Input) - Goods Issue",
                type=['csv'],
                key="upload_261",
                help="Raw materials consumed from stock"
            )

            file_101 = st.file_uploader(
                "101 (Output) - Goods Receipt",
                type=['csv'],
                key="upload_101",
                help="Finished goods received into stock"
            )

            # Upload button
            if st.button("Save Uploaded Files", type="primary", use_container_width=True):
                if not upload_year.isdigit() or len(upload_year) != 4:
                    st.error("Please enter a valid 4-digit year")
                elif not file_261 and not file_101:
                    st.error("Please upload at least one file")
                else:
                    import os
                    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                    os.makedirs(data_dir, exist_ok=True)

                    saved_files = []

                    if file_261:
                        path_261 = os.path.join(data_dir, f"261_{upload_year}.csv")
                        with open(path_261, "wb") as f:
                            f.write(file_261.getbuffer())
                        saved_files.append(f"261_{upload_year}.csv")

                    if file_101:
                        path_101 = os.path.join(data_dir, f"101_{upload_year}.csv")
                        with open(path_101, "wb") as f:
                            f.write(file_101.getbuffer())
                        saved_files.append(f"101_{upload_year}.csv")

                    st.success(f"Saved: {', '.join(saved_files)}")

                    # Check if both files now exist for this year
                    check_261 = os.path.exists(os.path.join(data_dir, f"261_{upload_year}.csv"))
                    check_101 = os.path.exists(os.path.join(data_dir, f"101_{upload_year}.csv"))

                    if check_261 and check_101:
                        st.success(f"Year {upload_year} is now ready for training!")
                    elif check_261:
                        st.warning(f"Still need 101_{upload_year}.csv to train for {upload_year}")
                    elif check_101:
                        st.warning(f"Still need 261_{upload_year}.csv to train for {upload_year}")

                    # Clear caches to detect new files
                    load_historical_data.clear()
                    st.rerun()

            # Show current data files status
            st.markdown("---")
            st.markdown("**Data Files Status:**")

            import os
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

            # Scan for all year files
            all_years = set()
            files_261 = set()
            files_101 = set()

            if os.path.exists(data_dir):
                import re
                for f in os.listdir(data_dir):
                    match = re.match(r'(101|261)_(\d{4})\.csv$', f, re.IGNORECASE)
                    if match:
                        file_type, year = match.groups()
                        all_years.add(year)
                        if file_type == '261':
                            files_261.add(year)
                        else:
                            files_101.add(year)

            if all_years:
                for year in sorted(all_years):
                    has_261 = year in files_261
                    has_101 = year in files_101
                    if has_261 and has_101:
                        st.success(f"{year}: Ready (261 + 101)")
                    elif has_261:
                        st.warning(f"{year}: Missing 101 file")
                    elif has_101:
                        st.warning(f"{year}: Missing 261 file")
            else:
                st.info("No data files uploaded yet")

        st.markdown("---")

        # =====================================================================
        # MODEL TRAINING SECTION
        # =====================================================================
        st.header("Model Training")

        # Get available years from data files
        available_years = get_available_years()

        if available_years:
            st.markdown(f"**Available years:** {', '.join(available_years)}")

            # Use checkboxes for year selection (clearer than multiselect)
            st.markdown("Select years to train:")
            selected_years = []
            cols = st.columns(len(available_years)) if len(available_years) <= 4 else st.columns(4)
            for i, year in enumerate(available_years):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.checkbox(year, value=True, key=f"year_{year}"):
                        selected_years.append(year)

            # Show selection summary
            if selected_years:
                st.info(f"Training on: {', '.join(selected_years)}")
            else:
                st.warning("Select at least one year")

            # Train button
            if st.button("Train Model", type="primary", use_container_width=True, disabled=len(selected_years) == 0):
                with st.spinner(f"Training model on {', '.join(selected_years)} data..."):
                    try:
                        # Prepare data for selected years
                        df, encoders = prepare_full_dataset(years=selected_years)

                        # Train model with year metadata
                        model = train_yield_model(df, encoders, years=selected_years)

                        st.success(f"Model trained successfully on {', '.join(selected_years)} data!")

                        # Clear cache to reload new model
                        load_model_artifacts.clear()
                        load_historical_data.clear()

                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("No data files found. Upload 261 and 101 CSV files above.")

        st.markdown("---")

        # =====================================================================
        # CURRENT MODEL INFO SECTION
        # =====================================================================
        st.header("Current Model")

        if artifacts:
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
            st.warning("No model trained yet")
            st.caption("Select years above and click 'Train Model' to get started")

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
        ["Forward Prediction (Input -> Output)", "Reverse Prediction (Output -> Input)"],
        horizontal=True,
        help="Forward: Calculate expected output from input materials. Reverse: Calculate required input for desired output."
    )

    st.markdown("---")

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
