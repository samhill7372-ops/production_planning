"""
Data Preparation Module for Material Yield Prediction System

SAP Manufacturing Logic:
- 261 = Goods Issue to Order = INPUT material consumption (raw materials CONSUMED from stock)
- 101 = Goods Receipt = OUTPUT material production (finished goods RECEIVED into stock)
- Input and output materials are DIFFERENT
- ONLY join key is MANUFACTURINGORDER (never match by MATERIAL)
- Yield = Total_Output_BF (from 101) / Total_Input_BF (from 261)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional, List
import warnings
import os

warnings.filterwarnings('ignore')

# Get the project root directory (parent of src/)
import re
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


def get_available_years() -> List[str]:
    """
    Scan data directory for year-specific CSV files.

    Looks for files matching patterns:
    - 101_YYYY.csv (output data)
    - 261_YYYY.csv (input data)

    Returns list of years where BOTH 101 and 261 files exist.
    """
    years_101 = set()
    years_261 = set()

    # Pattern to extract year from filename
    year_pattern = re.compile(r'(101|261)_(\d{4})\.csv$', re.IGNORECASE)

    # Scan data directory
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        filename = os.path.basename(filepath)
        match = year_pattern.match(filename)
        if match:
            file_type = match.group(1)
            year = match.group(2)
            if file_type == '101':
                years_101.add(year)
            elif file_type == '261':
                years_261.add(year)

    # Return years where both files exist
    common_years = years_101.intersection(years_261)
    return sorted(list(common_years))


def load_csv_files(
    input_261_path: str = None,
    output_101_path: str = None,
    years: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the 261 (input material) and 101 (output material) CSV files.

    SAP Logic:
    - 261.csv contains raw material INPUT consumption records (Goods Issue - material CONSUMED from stock)
    - 101.csv contains finished/semi-finished OUTPUT production records (Goods Receipt - material RECEIVED into stock)

    Args:
        input_261_path: Path to 261 CSV (used if years is None)
        output_101_path: Path to 101 CSV (used if years is None)
        years: List of years to load (e.g., ['2024', '2025']). If provided, loads year-specific files.

    Returns:
        Tuple of (df_261, df_101) DataFrames
    """
    # If years specified, load year-specific files
    if years is not None and len(years) > 0:
        df_261_list = []
        df_101_list = []

        for year in years:
            path_261 = os.path.join(DATA_DIR, f"261_{year}.csv")
            path_101 = os.path.join(DATA_DIR, f"101_{year}.csv")

            if os.path.exists(path_261):
                df = pd.read_csv(path_261)
                df['_source_year'] = year  # Track source year
                df_261_list.append(df)
                print(f"Loaded 261_{year}.csv (INPUT): {len(df):,} rows")
            else:
                print(f"WARNING: 261_{year}.csv not found")

            if os.path.exists(path_101):
                df = pd.read_csv(path_101)
                df['_source_year'] = year  # Track source year
                df_101_list.append(df)
                print(f"Loaded 101_{year}.csv (OUTPUT): {len(df):,} rows")
            else:
                print(f"WARNING: 101_{year}.csv not found")

        if not df_261_list or not df_101_list:
            raise ValueError(f"Could not load data for years: {years}")

        df_261 = pd.concat(df_261_list, ignore_index=True)
        df_101 = pd.concat(df_101_list, ignore_index=True)

        print(f"Combined 261 (INPUT - Goods Issue): {len(df_261):,} rows from {len(df_261_list)} year(s)")
        print(f"Combined 101 (OUTPUT - Goods Receipt): {len(df_101):,} rows from {len(df_101_list)} year(s)")

        return df_261, df_101

    # Fallback: Use default paths if not provided
    if input_261_path is None:
        input_261_path = os.path.join(DATA_DIR, "261.csv")
    if output_101_path is None:
        output_101_path = os.path.join(DATA_DIR, "101.csv")

    df_261 = pd.read_csv(input_261_path)
    print(f"Loaded 261.csv (INPUT - Goods Issue): {len(df_261):,} rows, {len(df_261.columns)} columns")

    df_101 = pd.read_csv(output_101_path)
    print(f"Loaded 101.csv (OUTPUT - Goods Receipt): {len(df_101):,} rows, {len(df_101.columns)} columns")

    return df_261, df_101


def load_raw_csv_data(years: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RAW 261 and 101 CSV data with minimal processing.

    This function loads the CSV files and only does:
    - Standardize column names (uppercase)
    - Convert MANUFACTURINGORDER to string
    - Convert BFIN/BFOUT to numeric

    It does NOT:
    - Remove IS_DELETED records
    - Aggregate data
    - Apply any other filtering

    Used for accurate historical totals in KD lookup to match notebook calculations.

    Returns:
        Tuple of (df_261_raw, df_101_raw) DataFrames
    """
    # Load CSV files
    df_261, df_101 = load_csv_files(years=years)

    # Minimal processing for 261
    df_261.columns = df_261.columns.str.upper().str.strip()
    if 'MANUFACTURINGORDER' in df_261.columns:
        df_261['MANUFACTURINGORDER'] = df_261['MANUFACTURINGORDER'].astype(str).str.strip()
    if 'BFIN' in df_261.columns:
        df_261['BFIN'] = pd.to_numeric(df_261['BFIN'], errors='coerce').fillna(0)
    if 'MATERIALTHICKNESS' in df_261.columns:
        df_261['MATERIALTHICKNESS'] = pd.to_numeric(df_261['MATERIALTHICKNESS'], errors='coerce').fillna(0)

    # Minimal processing for 101
    df_101.columns = df_101.columns.str.upper().str.strip()
    if 'MANUFACTURINGORDER' in df_101.columns:
        df_101['MANUFACTURINGORDER'] = df_101['MANUFACTURINGORDER'].astype(str).str.strip()
    if 'BFOUT' in df_101.columns:
        df_101['BFOUT'] = pd.to_numeric(df_101['BFOUT'], errors='coerce').fillna(0)

    print(f"Loaded RAW 261: {len(df_261):,} rows, BFIN total: {df_261['BFIN'].sum():,.0f}")
    print(f"Loaded RAW 101: {len(df_101):,} rows, BFOUT total: {df_101['BFOUT'].sum():,.0f}")

    return df_261, df_101


def clean_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Clean dataframe: standardize column names, convert types.
    """
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.upper().str.strip()

    # Convert MANUFACTURINGORDER to string for consistent joining
    if 'MANUFACTURINGORDER' in df.columns:
        df['MANUFACTURINGORDER'] = df['MANUFACTURINGORDER'].astype(str).str.strip()

    # Convert numeric columns to proper types (handle empty strings)
    numeric_columns = ['MATERIALTHICKNESS', 'TALLYLENGTH', 'TALLYWIDTH', 'BFIN', 'BFOUT']
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors (empty strings, etc.) to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove deleted records if flag exists
    if 'IS_DELETED' in df.columns:
        before = len(df)
        df = df[df['IS_DELETED'] == False]
        print(f"{df_name}: Removed {before - len(df)} deleted records")

    print(f"{df_name}: Cleaned, {len(df):,} records remaining")
    return df


def derive_bfout_from_dimensions(df_101: pd.DataFrame) -> pd.DataFrame:
    """
    Derive BFOUT (output board feet) from 101 data (Goods Receipt).

    Priority order:
    1. Use existing BFOUT column if it has valid values
    2. Use BFIN column if it has values (some SAP configs store output in BFIN)
    3. Derive from dimensions: (MATERIALTHICKNESS × TALLYWIDTH × TALLYLENGTH) / 144

    101 = Goods Receipt = finished goods RECEIVED into stock (OUTPUT)
    """
    df = df_101.copy()

    # Priority 1: Check if BFOUT column already exists with valid values
    if 'BFOUT' in df.columns and df['BFOUT'].sum() > 0:
        df['BFOUT'] = df['BFOUT'].fillna(0)
        print("Using existing BFOUT column for 101 data (Goods Receipt)")
    # Priority 2: Check if BFIN has values (some SAP configs use BFIN for output)
    elif 'BFIN' in df.columns and df['BFIN'].sum() > 0:
        df['BFOUT'] = df['BFIN'].fillna(0)
        print("Using BFIN column as BFOUT for 101 data (Goods Receipt)")
    # Priority 3: Derive from dimensions
    elif all(col in df.columns for col in ['MATERIALTHICKNESS', 'TALLYWIDTH', 'TALLYLENGTH']):
        df['BFOUT'] = (
            df['MATERIALTHICKNESS'].fillna(0) *
            df['TALLYWIDTH'].fillna(0) *
            df['TALLYLENGTH'].fillna(0)
        ) / 144
        print("Derived BFOUT from dimensions: (Thickness × Width × Length) / 144")
    else:
        raise ValueError("Cannot derive BFOUT: Missing required columns (BFOUT, BFIN, or dimension columns)")

    # Ensure non-negative
    df['BFOUT'] = df['BFOUT'].clip(lower=0)

    print(f"BFOUT: Mean={df['BFOUT'].mean():.2f}, Total={df['BFOUT'].sum():,.0f}")
    return df


def derive_bfin_from_dimensions(df_261: pd.DataFrame) -> pd.DataFrame:
    """
    Derive BFIN (input board feet) from 261 data (Goods Issue).

    Formula: BFIN = (MATERIALTHICKNESS × TALLYWIDTH × TALLYLENGTH) / 144
    All units are in inches.

    261 = Goods Issue to Order = raw materials CONSUMED from stock (INPUT)
    """
    df = df_261.copy()

    # Check if BFIN needs to be derived
    if 'BFIN' not in df.columns or df['BFIN'].sum() == 0:
        if all(col in df.columns for col in ['MATERIALTHICKNESS', 'TALLYWIDTH', 'TALLYLENGTH']):
            df['BFIN'] = (
                df['MATERIALTHICKNESS'].fillna(0) *
                df['TALLYWIDTH'].fillna(0) *
                df['TALLYLENGTH'].fillna(0)
            ) / 144
            print("Derived BFIN from dimensions: (Thickness × Width × Length) / 144")
            print(f"BFIN derived: Mean={df['BFIN'].mean():.2f}, Total={df['BFIN'].sum():,.0f}")
        else:
            raise ValueError("Cannot derive BFIN: Missing required dimension columns")

    # Ensure non-negative
    df['BFIN'] = df['BFIN'].clip(lower=0)

    return df


def aggregate_input_261(df_261: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate INPUT (261) data per MANUFACTURINGORDER.

    SAP Logic for 261 (Goods Issue to Order):
    - 261 = raw materials CONSUMED from stock (INPUT to production)
    - Sum BFIN → Total_Input_BF (derived from dimensions if needed)
    - First: MATERIAL (input material code), PLANT, SPECIE, GRADE
    - Mean: THICKNESS, LENGTH, WIDTH
    """
    df = df_261.copy()

    # Derive BFIN from dimensions if needed
    df = derive_bfin_from_dimensions(df)

    agg_dict = {}

    # Sum input board feet
    if 'BFIN' in df.columns:
        agg_dict['BFIN'] = 'sum'

    # Categorical - take first (primary input material)
    for col in ['MATERIAL', 'MATERIALSPECIE', 'TALLYGRADE', 'PLANT']:
        if col in df.columns:
            agg_dict[col] = 'first'

    # Numerical - take mean
    for col in ['MATERIALTHICKNESS', 'TALLYLENGTH', 'TALLYWIDTH']:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Perform aggregation
    df_agg = df.groupby('MANUFACTURINGORDER').agg(agg_dict).reset_index()

    # Rename BFIN to Total_Input_BF
    if 'BFIN' in df_agg.columns:
        df_agg.rename(columns={'BFIN': 'Total_Input_BF'}, inplace=True)

    # Add prefix to input columns to distinguish from output
    df_agg.rename(columns={
        'MATERIAL': 'Input_Material',
        'MATERIALSPECIE': 'Input_Specie',
        'TALLYGRADE': 'Input_Grade',
        'PLANT': 'Input_Plant',
        'MATERIALTHICKNESS': 'Input_Thickness',
        'TALLYLENGTH': 'Input_Length',
        'TALLYWIDTH': 'Input_Width'
    }, inplace=True)

    print(f"Aggregated 261 (INPUT - Goods Issue): {len(df_agg):,} manufacturing orders")
    if 'Total_Input_BF' in df_agg.columns:
        print(f"Total Input BF: {df_agg['Total_Input_BF'].sum():,.0f}")

    return df_agg


def aggregate_output_101(df_101: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate OUTPUT (101) data per MANUFACTURINGORDER and OUTPUT MATERIAL.

    SAP Logic for 101 (Goods Receipt):
    - 101 = finished goods RECEIVED into stock (OUTPUT from production)
    - Sum BFOUT per (order, material) → Total_Output_BF
    - Use DOMINANT grade (grade with highest BF) per material within each order
    - MULTI-OUTPUT MODEL: Keep ALL output materials per order (not just dominant)
    """
    df = df_101.copy()

    # Derive BFOUT first
    df = derive_bfout_from_dimensions(df)

    # Calculate dominant grade per (order, material) combination
    # This ensures we get the correct grade when a material has multiple grades
    def get_dominant_grade(group):
        """Return grade with highest total BFOUT in the group."""
        if 'TALLYGRADE' not in group.columns or 'BFOUT' not in group.columns:
            return group['TALLYGRADE'].iloc[0] if 'TALLYGRADE' in group.columns else 'N/A'
        grade_bf = group.groupby('TALLYGRADE')['BFOUT'].sum()
        return grade_bf.idxmax() if len(grade_bf) > 0 else group['TALLYGRADE'].iloc[0]

    # Group by (MANUFACTURINGORDER, MATERIAL) to keep all output materials
    # Then get dominant grade per material
    agg_dict = {
        'BFOUT': 'sum',
    }

    # Species - first (same material should have same species)
    if 'MATERIALSPECIE' in df.columns:
        agg_dict['MATERIALSPECIE'] = 'first'

    # Dimensions - mean
    for col in ['MATERIALTHICKNESS', 'TALLYLENGTH', 'TALLYWIDTH']:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Aggregate by (order, material)
    df_agg = df.groupby(['MANUFACTURINGORDER', 'MATERIAL']).agg(agg_dict).reset_index()

    # Get dominant grade per (order, material)
    dominant_grades = df.groupby(['MANUFACTURINGORDER', 'MATERIAL']).apply(
        get_dominant_grade, include_groups=False
    ).reset_index(name='Dominant_Grade')

    # Merge dominant grades
    df_agg = df_agg.merge(dominant_grades, on=['MANUFACTURINGORDER', 'MATERIAL'], how='left')

    # Rename for clarity
    df_agg.rename(columns={
        'BFOUT': 'Total_Output_BF',
        'MATERIAL': 'Output_Material',
        'MATERIALSPECIE': 'Output_Specie',
        'Dominant_Grade': 'Output_Grade',
        'MATERIALTHICKNESS': 'Output_Thickness',
        'TALLYLENGTH': 'Output_Length',
        'TALLYWIDTH': 'Output_Width'
    }, inplace=True)

    print(f"Aggregated 101 (OUTPUT - Goods Receipt): {len(df_agg):,} output records")
    print(f"Unique manufacturing orders: {df_agg['MANUFACTURINGORDER'].nunique():,}")
    print(f"Total Output BF: {df_agg['Total_Output_BF'].sum():,.0f}")

    return df_agg


def join_input_output_by_order(
    df_input: pd.DataFrame,
    df_output: pd.DataFrame
) -> pd.DataFrame:
    """
    CRITICAL: Join ONLY on MANUFACTURINGORDER.

    SAP Rule: Input and Output materials are DIFFERENT.
    Never join or match by MATERIAL column.
    """
    df_joined = pd.merge(
        df_input,
        df_output,
        on='MANUFACTURINGORDER',
        how='inner'
    )

    print(f"Joined data: {len(df_joined):,} orders with both input AND output")

    # Verify: Input_Material should be DIFFERENT from Output_Material
    if 'Input_Material' in df_joined.columns and 'Output_Material' in df_joined.columns:
        same_material = (df_joined['Input_Material'] == df_joined['Output_Material']).sum()
        if same_material > 0:
            print(f"WARNING: {same_material} orders have same input/output material (unusual)")

    return df_joined


def calculate_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Yield Percentage.

    Yield = (Total_Output_BF / Total_Input_BF) × 100
    """
    df = df.copy()

    df['Yield_Percentage'] = np.where(
        df['Total_Input_BF'] > 0,
        (df['Total_Output_BF'] / df['Total_Input_BF']) * 100,
        0
    )

    print(f"Yield Statistics:")
    print(f"  Mean: {df['Yield_Percentage'].mean():.2f}%")
    print(f"  Median: {df['Yield_Percentage'].median():.2f}%")
    print(f"  Min: {df['Yield_Percentage'].min():.2f}%")
    print(f"  Max: {df['Yield_Percentage'].max():.2f}%")

    return df


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    lower_pct: float = 1,
    upper_pct: float = 99
) -> pd.DataFrame:
    """Remove outliers using percentile bounds."""
    if column not in df.columns:
        return df

    lower = df[column].quantile(lower_pct / 100)
    upper = df[column].quantile(upper_pct / 100)

    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    removed = before - len(df)

    if removed > 0:
        print(f"Removed {removed} outliers from {column} (bounds: {lower:.2f} - {upper:.2f})")

    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical columns using LabelEncoder.
    """
    df = df.copy()

    if encoders is None:
        encoders = {}
        fit_new = True
    else:
        fit_new = False

    for col in columns:
        if col not in df.columns:
            continue

        # Fill missing and convert to string
        df[col] = df[col].fillna('Unknown').astype(str)
        encoded_col = f"{col}_Encoded"

        if fit_new:
            encoders[col] = LabelEncoder()
            df[encoded_col] = encoders[col].fit_transform(df[col])
            print(f"Encoded {col}: {len(encoders[col].classes_)} unique values")
        else:
            if col in encoders:
                le = encoders[col]
                # Handle unseen categories
                df[encoded_col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

    return df, encoders


def get_unique_materials_from_csv(
    input_261_path: str = None,
    output_101_path: str = None
) -> Dict[str, List[str]]:
    """
    Get all unique categorical values from CSV files or cached JSON.
    Returns dict with lists for dropdowns.

    SAP Logic:
    - 261 = Goods Issue = INPUT materials (raw materials consumed)
    - 101 = Goods Receipt = OUTPUT materials (finished goods produced)
    """
    import json

    # Use default paths if not provided
    if input_261_path is None:
        input_261_path = os.path.join(DATA_DIR, "261.csv")
    if output_101_path is None:
        output_101_path = os.path.join(DATA_DIR, "101.csv")

    result = {
        'Input_Material': [],
        'Input_Specie': [],
        'Input_Grade': [],
        'Input_Plant': [],
        'Output_Material': [],
        'Output_Specie': [],
        'Output_Grade': [],
        'Input_Thickness': []
    }

    # First, try to load from pre-computed JSON (for deployed environments)
    json_path = os.path.join(CONFIG_DIR, "dropdown_options.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                cached = json.load(f)
            result.update(cached)
            print(f"Loaded dropdown options from {json_path}")
            return result
        except Exception as e:
            print(f"Error loading cached options: {e}")

    # Fall back to loading from CSV files if JSON not available
    try:
        # Load 261 for INPUT materials (Goods Issue - raw materials consumed)
        # Only load required columns to save memory
        if os.path.exists(input_261_path):
            cols_needed_261 = ['MATERIAL', 'MATERIALSPECIE', 'TALLYGRADE', 'PLANT', 'MATERIALTHICKNESS']
            # Read header first to check column names
            header_df = pd.read_csv(input_261_path, nrows=0)
            header_df.columns = header_df.columns.str.upper().str.strip()
            available_cols = [c for c in cols_needed_261 if c in header_df.columns]

            if available_cols:
                df_261 = pd.read_csv(input_261_path, usecols=lambda x: x.upper().strip() in available_cols)
                df_261.columns = df_261.columns.str.upper().str.strip()

                if 'MATERIAL' in df_261.columns:
                    result['Input_Material'] = sorted(df_261['MATERIAL'].dropna().astype(str).unique().tolist())
                if 'MATERIALSPECIE' in df_261.columns:
                    result['Input_Specie'] = sorted(df_261['MATERIALSPECIE'].dropna().astype(str).unique().tolist())
                if 'TALLYGRADE' in df_261.columns:
                    result['Input_Grade'] = sorted(df_261['TALLYGRADE'].dropna().astype(str).unique().tolist())
                if 'PLANT' in df_261.columns:
                    result['Input_Plant'] = sorted(df_261['PLANT'].dropna().astype(str).unique().tolist())
                if 'MATERIALTHICKNESS' in df_261.columns:
                    result['Input_Thickness'] = sorted(df_261['MATERIALTHICKNESS'].dropna().unique().tolist())

                del df_261  # Free memory

        # Load 101 for OUTPUT materials (Goods Receipt - finished goods produced)
        # Only load required columns to save memory
        if os.path.exists(output_101_path):
            cols_needed_101 = ['MATERIAL', 'MATERIALSPECIE', 'TALLYGRADE']
            header_df = pd.read_csv(output_101_path, nrows=0)
            header_df.columns = header_df.columns.str.upper().str.strip()
            available_cols = [c for c in cols_needed_101 if c in header_df.columns]

            if available_cols:
                df_101 = pd.read_csv(output_101_path, usecols=lambda x: x.upper().strip() in available_cols)
                df_101.columns = df_101.columns.str.upper().str.strip()

                if 'MATERIAL' in df_101.columns:
                    result['Output_Material'] = sorted(df_101['MATERIAL'].dropna().astype(str).unique().tolist())
                if 'MATERIALSPECIE' in df_101.columns:
                    result['Output_Specie'] = sorted(df_101['MATERIALSPECIE'].dropna().astype(str).unique().tolist())
                if 'TALLYGRADE' in df_101.columns:
                    result['Output_Grade'] = sorted(df_101['TALLYGRADE'].dropna().astype(str).unique().tolist())

                del df_101  # Free memory

    except Exception as e:
        print(f"Error loading materials: {e}")

    return result


def get_historical_yield_by_material(
    df: pd.DataFrame,
    input_material: str
) -> pd.DataFrame:
    """
    Get historical yield data for a specific input material.
    Returns possible output materials and their yield statistics.
    """
    if 'Input_Material' not in df.columns:
        return pd.DataFrame()

    subset = df[df['Input_Material'] == input_material]

    if len(subset) == 0:
        return pd.DataFrame()

    # Group by output material
    yield_stats = subset.groupby('Output_Material').agg({
        'Yield_Percentage': ['mean', 'std', 'min', 'max', 'count'],
        'Total_Output_BF': 'sum',
        'Total_Input_BF': 'sum'
    }).reset_index()

    yield_stats.columns = [
        'Output_Material',
        'Mean_Yield', 'Std_Yield', 'Min_Yield', 'Max_Yield', 'Order_Count',
        'Total_Output', 'Total_Input'
    ]

    yield_stats = yield_stats.sort_values('Mean_Yield', ascending=False)

    return yield_stats


def prepare_full_dataset(
    input_261_path: str = None,
    output_101_path: str = None,
    remove_yield_outliers: bool = True,
    years: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Complete data preparation pipeline following SAP logic.

    SAP Movement Types:
    - 261 = Goods Issue to Order = INPUT (raw materials CONSUMED from stock)
    - 101 = Goods Receipt = OUTPUT (finished goods RECEIVED into stock)
    """
    print("=" * 70)
    print("DATA PREPARATION PIPELINE (SAP Manufacturing Logic)")
    print("=" * 70)
    print("Movement Type 261 = Goods Issue = INPUT (raw materials consumed)")
    print("Movement Type 101 = Goods Receipt = OUTPUT (finished goods produced)")

    # Step 1: Load CSV files
    print("\n[1] Loading CSV files...")
    if years:
        print(f"Loading data for years: {years}")
    df_261, df_101 = load_csv_files(input_261_path, output_101_path, years=years)

    # Step 2: Clean dataframes
    print("\n[2] Cleaning dataframes...")
    df_261 = clean_dataframe(df_261, "261 (INPUT - Goods Issue)")
    df_101 = clean_dataframe(df_101, "101 (OUTPUT - Goods Receipt)")

    # Step 3: Aggregate INPUT per manufacturing order (261 = Goods Issue)
    print("\n[3] Aggregating INPUT (261 - Goods Issue) per manufacturing order...")
    df_input_agg = aggregate_input_261(df_261)

    # Step 4: Aggregate OUTPUT per manufacturing order (101 = Goods Receipt)
    print("\n[4] Aggregating OUTPUT (101 - Goods Receipt) per manufacturing order...")
    df_output_agg = aggregate_output_101(df_101)

    # Step 5: Join ONLY on MANUFACTURINGORDER
    print("\n[5] Joining on MANUFACTURINGORDER...")
    df_joined = join_input_output_by_order(df_input_agg, df_output_agg)

    # Step 6: Calculate Yield
    print("\n[6] Calculating Yield Percentage...")
    df_joined = calculate_yield(df_joined)

    # Step 7: Remove outliers
    if remove_yield_outliers:
        print("\n[7] Removing outliers...")
        df_joined = remove_outliers(df_joined, 'Yield_Percentage', 1, 99)
        df_joined = remove_outliers(df_joined, 'Total_Input_BF', 1, 99)
        df_joined = remove_outliers(df_joined, 'Total_Output_BF', 1, 99)

    # Step 8: Encode categorical features
    print("\n[8] Encoding categorical features...")
    categorical_cols = ['Input_Material', 'Input_Specie', 'Input_Grade', 'Input_Plant',
                        'Output_Material', 'Output_Specie', 'Output_Grade']
    categorical_cols = [c for c in categorical_cols if c in df_joined.columns]

    df_encoded, encoders = encode_categorical(df_joined, categorical_cols)

    # Fill any remaining missing values
    for col in df_encoded.select_dtypes(include=[np.number]).columns:
        if df_encoded[col].isnull().any():
            df_encoded[col].fillna(df_encoded[col].median(), inplace=True)

    print("\n" + "=" * 70)
    print(f"PREPARATION COMPLETE: {len(df_encoded):,} records, {len(df_encoded.columns)} columns")
    print("=" * 70)

    return df_encoded, encoders


def prepare_full_dataset_with_raw(
    input_261_path: str = None,
    output_101_path: str = None,
    remove_yield_outliers: bool = True,
    years: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Complete data preparation pipeline that ALSO returns raw aggregated dataframes.

    This variant returns the pre-join aggregated data (df_input_agg, df_output_agg)
    in addition to the joined data. Useful for calculating totals that include
    orders without matches (e.g., KD lookup historical totals).

    Returns:
        Tuple of (df_joined, df_input_agg, df_output_agg, encoders)
        - df_joined: Joined and encoded data (same as prepare_full_dataset)
        - df_input_agg: Aggregated 261 data before join (all input orders)
        - df_output_agg: Aggregated 101 data before join (all output orders)
        - encoders: Label encoders for categorical columns
    """
    print("=" * 70)
    print("DATA PREPARATION PIPELINE WITH RAW (SAP Manufacturing Logic)")
    print("=" * 70)

    # Step 1: Load CSV files
    print("\n[1] Loading CSV files...")
    if years:
        print(f"Loading data for years: {years}")
    df_261, df_101 = load_csv_files(input_261_path, output_101_path, years=years)

    # Step 2: Clean dataframes
    print("\n[2] Cleaning dataframes...")
    df_261 = clean_dataframe(df_261, "261 (INPUT - Goods Issue)")
    df_101 = clean_dataframe(df_101, "101 (OUTPUT - Goods Receipt)")

    # Step 3: Aggregate INPUT per manufacturing order (261 = Goods Issue)
    print("\n[3] Aggregating INPUT (261 - Goods Issue) per manufacturing order...")
    df_input_agg = aggregate_input_261(df_261)

    # Step 4: Aggregate OUTPUT per manufacturing order (101 = Goods Receipt)
    print("\n[4] Aggregating OUTPUT (101 - Goods Receipt) per manufacturing order...")
    df_output_agg = aggregate_output_101(df_101)

    # Step 5: Join ONLY on MANUFACTURINGORDER
    print("\n[5] Joining on MANUFACTURINGORDER...")
    df_joined = join_input_output_by_order(df_input_agg, df_output_agg)

    # Step 6: Calculate Yield
    print("\n[6] Calculating Yield Percentage...")
    df_joined = calculate_yield(df_joined)

    # Step 7: Remove outliers (only from joined data, not raw)
    if remove_yield_outliers:
        print("\n[7] Removing outliers...")
        df_joined = remove_outliers(df_joined, 'Yield_Percentage', 1, 99)
        df_joined = remove_outliers(df_joined, 'Total_Input_BF', 1, 99)
        df_joined = remove_outliers(df_joined, 'Total_Output_BF', 1, 99)

    # Step 8: Encode categorical features
    print("\n[8] Encoding categorical features...")
    categorical_cols = ['Input_Material', 'Input_Specie', 'Input_Grade', 'Input_Plant',
                        'Output_Material', 'Output_Specie', 'Output_Grade']
    categorical_cols = [c for c in categorical_cols if c in df_joined.columns]

    df_encoded, encoders = encode_categorical(df_joined, categorical_cols)

    # Fill any remaining missing values
    for col in df_encoded.select_dtypes(include=[np.number]).columns:
        if df_encoded[col].isnull().any():
            df_encoded[col].fillna(df_encoded[col].median(), inplace=True)

    print("\n" + "=" * 70)
    print(f"PREPARATION COMPLETE: {len(df_encoded):,} joined records")
    print(f"Raw 261 (input): {len(df_input_agg):,} orders")
    print(f"Raw 101 (output): {len(df_output_agg):,} records")
    print("=" * 70)

    return df_encoded, df_input_agg, df_output_agg, encoders


def get_feature_columns() -> List[str]:
    """Feature columns for yield prediction model."""
    return [
        'Input_Material_Encoded',
        'Input_Specie_Encoded',
        'Input_Grade_Encoded',
        'Input_Plant_Encoded',
        'Input_Thickness',
        'Input_Length',
        'Input_Width',
        'Total_Input_BF'
    ]


if __name__ == "__main__":
    csv_261 = os.path.join(DATA_DIR, "261.csv")
    csv_101 = os.path.join(DATA_DIR, "101.csv")
    if os.path.exists(csv_261) and os.path.exists(csv_101):
        df, encoders = prepare_full_dataset(csv_261, csv_101)
        print("\nSample data:")
        print(df.head())
        print("\nColumns:", list(df.columns))
    else:
        print(f"CSV files not found in {DATA_DIR} (need both 261.csv and 101.csv).")
