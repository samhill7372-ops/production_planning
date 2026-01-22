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
    get_historical_yield_by_material
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
    try:
        model_path = os.path.join(MODELS_DIR, "yield_model.joblib")
        if not os.path.exists(model_path):
            return None

        model = joblib.load(model_path)
        encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "encoders.joblib")) else {}
        feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "feature_columns.joblib")) else []
        metrics = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "metrics.joblib")) else {}
        test_results = pd.read_csv(os.path.join(DATA_DIR, "test_results.csv")) if os.path.exists(os.path.join(DATA_DIR, "test_results.csv")) else None

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
            'classifier_metrics': classifier_metrics
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
        st.markdown(f"""
        **How we calculated this:**

        1. **Historical Data**: Found {recommendation['historical_orders']} past orders
           with similar input materials producing this output.

        2. **Yield Analysis**: Historical average yield combined with ML model prediction
           gives expected yield of {recommendation['expected_yield']:.1f}%.

        3. **Confidence Range**: The yield range ({recommendation['yield_range']})
           represents a 95% confidence interval based on historical variation.

        4. **Risk Assessment**: {recommendation['risk_level']} risk because:
           - {'Sufficient' if recommendation['historical_orders'] >= 20 else 'Limited'} historical data
           - {'Consistent' if recommendation['confidence_level'] in ['HIGH', 'MEDIUM'] else 'Variable'} yield patterns

        **What this means for planning:**
        - Plan for **{recommendation['expected_output_bf']:,.0f} BF** output (best estimate)
        - Worst case: **{recommendation['output_range_bf'].split(' - ')[0]}**
        - Best case: **{recommendation['output_range_bf'].split(' - ')[1]}**
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
        st.header("System Info")

        if artifacts:
            st.success("Yield Model Loaded")
            if artifacts['metrics']:
                best_r2 = max(
                    m.get('test_r2', m.get('R2', 0))
                    for m in artifacts['metrics'].values()
                    if isinstance(m, dict)
                )
                st.info(f"Yield Model R²: {best_r2:.4f}")

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
                    st.info(f"Classifier Accuracy: {best_acc:.2%}")
            else:
                st.warning("Output Classifier not trained")
        else:
            st.warning("Model not trained")
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        df, encoders = prepare_full_dataset()
                        model = train_yield_model(df, encoders)
                        st.success("Model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

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
                    st.session_state.prediction_run = True

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display Results
    if st.session_state.get('prediction_run'):
        st.markdown("---")
        st.header("3. Prediction Results")

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Enhanced Results",
            "Analysis Chart",
            "Basic Results",
            "Model Performance",
            "Test Set Results"
        ])

        with tab1:
            render_enhanced_results(output_results)

        with tab2:
            render_analysis_chart(output_results)

        with tab3:
            render_detailed_results(output_results)

        with tab4:
            render_model_performance(metrics)

        with tab5:
            render_test_results(test_results)


if __name__ == "__main__":
    main()
