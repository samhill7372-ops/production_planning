"""
Deployment Preparation Script

Pre-computes all historical statistics from CSV files and saves them as compact
joblib files. These files replace the need for large CSV files at runtime.

Run this script BEFORE deploying to BTP Cloud Foundry:
    python scripts/prepare_deployment.py

This generates:
    models/YYYY/historical_data_precomputed.joblib (~2-5 MB per year)

The app will use these pre-computed files when CSV files are not available.
"""

import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import joblib

from src.data_preparation import (
    load_raw_csv_data,
    get_available_years,
    prepare_full_dataset
)


def compute_material_stats(
    df_261_raw: pd.DataFrame,
    df_101_raw: pd.DataFrame,
    min_order_count: int = 5
) -> dict:
    """
    Pre-compute material-level statistics for forward prediction.

    Structure:
    {
        "KS_MATERIAL": {
            "PLANT": {
                "total_hist_input_bf": float,
                "total_hist_output_bf": float,
                "historical_yield_pct": float,
                "total_orders": int,
                "avg_thickness": float,
                "avg_length": float,
                "avg_width": float,
                "most_common_specie": str,
                "most_common_grade": str,
                "kd_distribution": [
                    {
                        "KD_Material": str,
                        "Historical_BF_Output": float,
                        "Order_Count": int,
                        "Contribution_Pct": float
                    },
                    ...
                ]
            }
        }
    }
    """
    material_stats = {}

    # Get unique (Material, Plant) combinations from 261
    unique_combos = df_261_raw.groupby(['MATERIAL', 'PLANT']).size().reset_index(name='count')
    total_combos = len(unique_combos)

    print(f"Processing {total_combos} (Material, Plant) combinations...")

    for idx, row in unique_combos.iterrows():
        ks_material = row['MATERIAL']
        plant = row['PLANT']

        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total_combos}")

        # Filter 261 data
        filtered_261 = df_261_raw[
            (df_261_raw['MATERIAL'] == ks_material) &
            (df_261_raw['PLANT'] == plant)
        ]

        if len(filtered_261) == 0:
            continue

        # Get manufacturing orders
        order_list = filtered_261['MANUFACTURINGORDER'].unique()
        total_orders = len(order_list)

        # Skip if too few orders
        if total_orders < min_order_count:
            continue

        # Calculate input totals
        total_hist_input_bf = float(filtered_261['BFIN'].sum())

        # Get average dimensions (handle mixed types by converting to numeric)
        avg_thickness = 4.0
        avg_length = 96.0
        avg_width = 8.0

        if 'MATERIALTHICKNESS' in filtered_261.columns:
            thickness_vals = pd.to_numeric(filtered_261['MATERIALTHICKNESS'], errors='coerce')
            if thickness_vals.notna().any():
                avg_thickness = float(thickness_vals.mean())

        if 'TALLYLENGTH' in filtered_261.columns:
            length_vals = pd.to_numeric(filtered_261['TALLYLENGTH'], errors='coerce')
            if length_vals.notna().any():
                avg_length = float(length_vals.mean())

        if 'TALLYWIDTH' in filtered_261.columns:
            width_vals = pd.to_numeric(filtered_261['TALLYWIDTH'], errors='coerce')
            if width_vals.notna().any():
                avg_width = float(width_vals.mean())

        # Get most common specie and grade
        most_common_specie = 'SM'
        if 'MATERIALSPECIE' in filtered_261.columns and len(filtered_261['MATERIALSPECIE'].dropna()) > 0:
            specie_mode = filtered_261['MATERIALSPECIE'].mode()
            if len(specie_mode) > 0:
                most_common_specie = str(specie_mode.iloc[0])

        most_common_grade = '2C'
        if 'TALLYGRADE' in filtered_261.columns and len(filtered_261['TALLYGRADE'].dropna()) > 0:
            grade_mode = filtered_261['TALLYGRADE'].mode()
            if len(grade_mode) > 0:
                most_common_grade = str(grade_mode.iloc[0])

        # Get output data for these orders
        filtered_101 = df_101_raw[df_101_raw['MANUFACTURINGORDER'].isin(order_list)]

        if len(filtered_101) == 0:
            continue

        total_hist_output_bf = float(filtered_101['BFOUT'].sum())
        historical_yield_pct = (total_hist_output_bf / total_hist_input_bf * 100) if total_hist_input_bf > 0 else 0

        # Calculate KD distribution
        output_agg = filtered_101.groupby('MATERIAL').agg({
            'BFOUT': 'sum',
            'MANUFACTURINGORDER': 'nunique'
        }).reset_index()
        output_agg.columns = ['KD_Material', 'Historical_BF_Output', 'Order_Count']

        # Filter by minimum order count
        output_agg = output_agg[output_agg['Order_Count'] >= min_order_count]

        if len(output_agg) == 0:
            continue

        # Calculate contribution percentage
        total_filtered_bf = output_agg['Historical_BF_Output'].sum()
        output_agg['Contribution_Pct'] = (output_agg['Historical_BF_Output'] / total_filtered_bf * 100).round(2)
        output_agg = output_agg.sort_values('Contribution_Pct', ascending=False)

        # Convert to list of dicts
        kd_distribution = output_agg.to_dict('records')

        # Store in nested structure
        if ks_material not in material_stats:
            material_stats[ks_material] = {}

        material_stats[ks_material][plant] = {
            'total_hist_input_bf': round(total_hist_input_bf, 2),
            'total_hist_output_bf': round(total_hist_output_bf, 2),
            'historical_yield_pct': round(historical_yield_pct, 2),
            'total_orders': total_orders,
            'avg_thickness': round(avg_thickness, 2),
            'avg_length': round(avg_length, 2),
            'avg_width': round(avg_width, 2),
            'most_common_specie': most_common_specie,
            'most_common_grade': most_common_grade,
            'kd_distribution': kd_distribution
        }

    print(f"Computed stats for {len(material_stats)} unique materials")
    return material_stats


def compute_kd_lookup_distribution(
    historical_data: pd.DataFrame
) -> dict:
    """
    Pre-compute KD distribution lookup for all input materials.

    Structure:
    {
        "INPUT_MATERIAL": {
            "kd_outputs": [
                {
                    "Output_Material": str,
                    "Output_Grade": str,
                    "BF_Output": float,
                    "Order_Count": int,
                    "Percentage": float,
                    "Avg_Yield": float,
                    "Yield_Std": float
                },
                ...
            ],
            "total_bf_output": float,
            "total_orders": int
        }
    }
    """
    kd_lookup = {}

    if historical_data is None or len(historical_data) == 0:
        return kd_lookup

    if 'Input_Material' not in historical_data.columns or 'Output_Material' not in historical_data.columns:
        return kd_lookup

    unique_inputs = historical_data['Input_Material'].unique()
    print(f"Processing KD lookup for {len(unique_inputs)} input materials...")

    has_output_grade = 'Output_Grade' in historical_data.columns
    has_yield = 'Yield_Percentage' in historical_data.columns
    has_bf_output = 'Total_Output_BF' in historical_data.columns

    for idx, input_material in enumerate(unique_inputs):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(unique_inputs)}")

        filtered = historical_data[historical_data['Input_Material'] == input_material].copy()

        if len(filtered) == 0:
            continue

        # Group columns
        group_cols = ['Output_Material']
        if has_output_grade:
            group_cols.append('Output_Grade')

        # Build aggregation
        if has_yield:
            agg_cols = {}
            if has_bf_output:
                agg_cols['Total_Output_BF'] = 'sum'
            agg_cols['Yield_Percentage'] = ['mean', 'std', 'count']

            output_stats = filtered.groupby(group_cols).agg(agg_cols).reset_index()

            # Flatten column names
            new_cols = list(group_cols)
            if has_bf_output:
                new_cols.append('BF_Output')
            new_cols.extend(['Avg_Yield', 'Yield_Std', 'Order_Count'])
            output_stats.columns = new_cols

            output_stats['Yield_Std'] = output_stats['Yield_Std'].fillna(0)
            output_stats['Avg_Yield'] = output_stats['Avg_Yield'].round(2)
            output_stats['Yield_Std'] = output_stats['Yield_Std'].round(2)
            if has_bf_output:
                output_stats['BF_Output'] = output_stats['BF_Output'].round(2)
        else:
            if has_bf_output:
                output_stats = filtered.groupby(group_cols).agg({
                    'Total_Output_BF': 'sum'
                }).reset_index()
                output_stats.rename(columns={'Total_Output_BF': 'BF_Output'}, inplace=True)
                output_stats['BF_Output'] = output_stats['BF_Output'].round(2)
                output_stats['Order_Count'] = filtered.groupby(group_cols).size().values
            else:
                output_stats = filtered.groupby(group_cols).size().reset_index(name='Order_Count')
                output_stats['BF_Output'] = 0.0
            output_stats['Avg_Yield'] = 0.0
            output_stats['Yield_Std'] = 0.0

        # Add BF_Output if missing
        if 'BF_Output' not in output_stats.columns:
            output_stats['BF_Output'] = 0.0

        # Add Output_Grade if missing
        if not has_output_grade:
            output_stats['Output_Grade'] = 'N/A'

        # Calculate percentages
        total_bf = output_stats['BF_Output'].sum()
        if total_bf > 0:
            output_stats['Percentage'] = (output_stats['BF_Output'] / total_bf * 100).round(2)
        else:
            total_orders = output_stats['Order_Count'].sum()
            output_stats['Percentage'] = (output_stats['Order_Count'] / total_orders * 100).round(2) if total_orders > 0 else 0.0

        # Sort
        output_stats = output_stats.sort_values(
            ['Output_Material', 'BF_Output'],
            ascending=[True, False]
        ).reset_index(drop=True)

        # Ensure column order
        cols = ['Output_Material', 'Output_Grade', 'BF_Output', 'Order_Count', 'Percentage', 'Avg_Yield', 'Yield_Std']
        output_stats = output_stats[[c for c in cols if c in output_stats.columns]]

        kd_lookup[input_material] = {
            'kd_outputs': output_stats.to_dict('records'),
            'total_bf_output': round(total_bf, 2),
            'total_orders': int(output_stats['Order_Count'].sum())
        }

    print(f"Computed KD lookup for {len(kd_lookup)} input materials")
    return kd_lookup


def compute_historical_summary(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame with aggregated yield statistics.

    This is used as a fallback when more detailed lookups aren't available.
    """
    if historical_data is None or len(historical_data) == 0:
        return pd.DataFrame()

    # Group by key material attributes
    group_cols = ['Input_Material', 'Input_Plant', 'Input_Thickness', 'Input_Grade', 'Input_Specie',
                  'Output_Material', 'Output_Grade']
    group_cols = [c for c in group_cols if c in historical_data.columns]

    if len(group_cols) == 0:
        return pd.DataFrame()

    agg_dict = {}
    if 'Yield_Percentage' in historical_data.columns:
        agg_dict['Yield_Percentage'] = ['mean', 'std', 'min', 'max', 'count']
    if 'Total_Input_BF' in historical_data.columns:
        agg_dict['Total_Input_BF'] = 'sum'
    if 'Total_Output_BF' in historical_data.columns:
        agg_dict['Total_Output_BF'] = 'sum'

    if len(agg_dict) == 0:
        return pd.DataFrame()

    summary = historical_data.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    new_cols = list(group_cols)
    if 'Yield_Percentage' in agg_dict:
        new_cols.extend(['Mean_Yield', 'Std_Yield', 'Min_Yield', 'Max_Yield', 'Order_Count'])
    if 'Total_Input_BF' in agg_dict:
        new_cols.append('Total_Input_BF')
    if 'Total_Output_BF' in agg_dict:
        new_cols.append('Total_Output_BF')

    summary.columns = new_cols

    print(f"Created historical summary with {len(summary)} records")
    return summary


def compute_grade_breakdown_data(
    df_261_raw: pd.DataFrame,
    df_101_raw: pd.DataFrame,
    min_order_count: int = 3
) -> dict:
    """
    Pre-compute grade-level breakdown for each (KS_Material, Plant, KD_Material) combination.

    This enables grade breakdown display in deployment mode without raw CSV files.
    Uses memory-efficient approach by first aggregating data before joining.

    Structure:
    {
        "KS_MATERIAL": {
            "PLANT": {
                "KD_MATERIAL": [
                    {
                        "Grade": "PR",
                        "Order_Count": 5,
                        "BF_Output": 3200.0,
                        "Pct_Of_KD": 33.4,
                        "Avg_Length": 96.0,
                        "Avg_Width": 8.0,
                        "Avg_Thickness": 4.0
                    },
                    ...
                ]
            }
        }
    }
    """
    grade_breakdown = {}

    if df_261_raw is None or df_101_raw is None:
        return grade_breakdown

    if len(df_261_raw) == 0 or len(df_101_raw) == 0:
        return grade_breakdown

    print("Computing grade breakdown data (memory-efficient)...")

    # Step 1: Find common manufacturing orders
    print("  Finding common manufacturing orders...")
    orders_261 = set(df_261_raw['MANUFACTURINGORDER'].unique())
    orders_101 = set(df_101_raw['MANUFACTURINGORDER'].unique())
    common_orders = orders_261 & orders_101
    print(f"  Common orders: {len(common_orders):,}")

    if len(common_orders) == 0:
        return grade_breakdown

    # Step 2: Filter to common orders only (reduces data size significantly)
    print("  Filtering to common orders...")
    df_261 = df_261_raw[df_261_raw['MANUFACTURINGORDER'].isin(common_orders)].copy()
    df_101 = df_101_raw[df_101_raw['MANUFACTURINGORDER'].isin(common_orders)].copy()

    print(f"  Filtered 261: {len(df_261):,} rows, 101: {len(df_101):,} rows")

    # Normalize grade column
    if 'TALLYGRADE' in df_261.columns:
        df_261['TALLYGRADE'] = df_261['TALLYGRADE'].astype(str).str.upper().str.strip()

    # Convert dimensions to numeric
    dim_cols = ['TALLYLENGTH', 'TALLYWIDTH', 'MATERIALTHICKNESS']
    for col in dim_cols:
        if col in df_261.columns:
            df_261[col] = pd.to_numeric(df_261[col], errors='coerce')

    # Step 3: Aggregate 261 data by (Order, Material, Plant, Grade) first
    print("  Aggregating input data...")
    group_cols_261 = ['MANUFACTURINGORDER', 'MATERIAL', 'PLANT']
    if 'TALLYGRADE' in df_261.columns:
        group_cols_261.append('TALLYGRADE')

    agg_261 = {
        'BFIN': 'sum'
    }
    for col in dim_cols:
        if col in df_261.columns:
            agg_261[col] = 'mean'

    df_261_agg = df_261.groupby(group_cols_261).agg(agg_261).reset_index()

    # Step 4: Aggregate 101 data by (Order, Material) first
    print("  Aggregating output data...")
    df_101_agg = df_101.groupby(['MANUFACTURINGORDER', 'MATERIAL']).agg({
        'BFOUT': 'sum'
    }).reset_index()

    # Step 5: Join aggregated data
    print("  Joining aggregated data...")
    merged = df_261_agg.merge(
        df_101_agg,
        on='MANUFACTURINGORDER',
        suffixes=('_IN', '_OUT')
    )

    if len(merged) == 0:
        print("  No matching data found")
        return grade_breakdown

    # Rename columns
    merged = merged.rename(columns={
        'MATERIAL_IN': 'KS_Material',
        'MATERIAL_OUT': 'KD_Material',
        'PLANT': 'Plant',
        'TALLYGRADE': 'Grade'
    })

    if 'Grade' not in merged.columns:
        merged['Grade'] = 'N/A'

    print(f"  Merged data: {len(merged):,} records")

    # Step 6: Final aggregation by (KS, Plant, KD, Grade)
    print("  Final aggregation...")
    group_cols = ['KS_Material', 'Plant', 'KD_Material', 'Grade']

    agg_dict = {
        'MANUFACTURINGORDER': 'nunique',
        'BFOUT': 'sum'
    }

    # Add dimension columns if present
    for col in dim_cols:
        if col in merged.columns:
            agg_dict[col] = 'mean'

    agg_data = merged.groupby(group_cols).agg(agg_dict).reset_index()

    # Rename columns
    rename_dict = {
        'MANUFACTURINGORDER': 'Order_Count',
        'BFOUT': 'BF_Output'
    }
    for col in dim_cols:
        short_name = col.replace('TALLY', 'Avg_').replace('MATERIALTHICKNESS', 'Avg_Thickness')
        if col in agg_data.columns:
            rename_dict[col] = short_name
    agg_data = agg_data.rename(columns=rename_dict)

    # Step 7: Filter by minimum order count (at KS+Plant level)
    ks_plant_orders = agg_data.groupby(['KS_Material', 'Plant'])['Order_Count'].sum().reset_index()
    ks_plant_orders = ks_plant_orders[ks_plant_orders['Order_Count'] >= min_order_count]
    valid_combos = set(zip(ks_plant_orders['KS_Material'], ks_plant_orders['Plant']))
    agg_data = agg_data[agg_data.apply(lambda r: (r['KS_Material'], r['Plant']) in valid_combos, axis=1)]

    print(f"  After filtering: {len(agg_data):,} grade entries")

    # Step 8: Calculate percentages within each KD material
    kd_totals = agg_data.groupby(['KS_Material', 'Plant', 'KD_Material'])['BF_Output'].sum().reset_index()
    kd_totals = kd_totals.rename(columns={'BF_Output': 'KD_Total_BF'})
    agg_data = agg_data.merge(kd_totals, on=['KS_Material', 'Plant', 'KD_Material'])
    agg_data['Pct_Of_KD'] = (agg_data['BF_Output'] / agg_data['KD_Total_BF'] * 100).round(1)

    # Step 9: Build nested dictionary
    print("  Building nested structure...")
    for _, row in agg_data.iterrows():
        ks = row['KS_Material']
        plant = row['Plant']
        kd = row['KD_Material']

        if ks not in grade_breakdown:
            grade_breakdown[ks] = {}
        if plant not in grade_breakdown[ks]:
            grade_breakdown[ks][plant] = {}
        if kd not in grade_breakdown[ks][plant]:
            grade_breakdown[ks][plant][kd] = []

        entry = {
            'Grade': str(row['Grade']),
            'Order_Count': int(row['Order_Count']),
            'BF_Output': round(float(row['BF_Output']), 2),
            'Pct_Of_KD': float(row['Pct_Of_KD'])
        }

        if 'Avg_LENGTH' in row and pd.notna(row.get('Avg_LENGTH')):
            entry['Avg_Length'] = round(float(row['Avg_LENGTH']), 1)
        if 'Avg_WIDTH' in row and pd.notna(row.get('Avg_WIDTH')):
            entry['Avg_Width'] = round(float(row['Avg_WIDTH']), 1)
        if 'Avg_Thickness' in row and pd.notna(row.get('Avg_Thickness')):
            entry['Avg_Thickness'] = round(float(row['Avg_Thickness']), 2)

        grade_breakdown[ks][plant][kd].append(entry)

    # Sort grades by BF_Output within each KD
    for ks in grade_breakdown:
        for plant in grade_breakdown[ks]:
            for kd in grade_breakdown[ks][plant]:
                grade_breakdown[ks][plant][kd].sort(key=lambda x: x['BF_Output'], reverse=True)

    # Count total entries
    total_entries = sum(
        len(kd_dict)
        for plant_dict in grade_breakdown.values()
        for kd_dict in plant_dict.values()
    )
    print(f"  Computed grade breakdown for {len(grade_breakdown)} materials, {total_entries} total KD entries")

    return grade_breakdown


def prepare_deployment_for_year(year: str, models_dir: str, data_dir: str):
    """
    Prepare all pre-computed data for a specific year.
    """
    print(f"\n{'='*70}")
    print(f"PREPARING DEPLOYMENT DATA FOR YEAR: {year}")
    print(f"{'='*70}")

    # Load raw CSV data
    print("\n[1] Loading raw CSV data...")
    try:
        df_261_raw, df_101_raw = load_raw_csv_data(years=[year])
    except Exception as e:
        print(f"ERROR: Could not load CSV data for {year}: {e}")
        return False

    # Load historical data (joined + encoded)
    print("\n[2] Preparing historical data...")
    try:
        historical_data, encoders = prepare_full_dataset(years=[year])
    except Exception as e:
        print(f"ERROR: Could not prepare historical data for {year}: {e}")
        return False

    # Compute material stats
    print("\n[3] Computing material-level statistics...")
    material_stats = compute_material_stats(df_261_raw, df_101_raw)

    # Compute KD lookup
    print("\n[4] Computing KD distribution lookup...")
    kd_lookup = compute_kd_lookup_distribution(historical_data)

    # Compute historical summary
    print("\n[5] Computing historical summary...")
    summary_df = compute_historical_summary(historical_data)

    # Compute grade breakdown data (for deployment mode)
    print("\n[6] Computing grade breakdown data...")
    grade_breakdown_data = compute_grade_breakdown_data(df_261_raw, df_101_raw)

    # Package everything
    precomputed = {
        'material_stats': material_stats,
        'kd_lookup': kd_lookup,
        'grade_breakdown_data': grade_breakdown_data,
        'summary_df': summary_df,
        'generated_at': datetime.now().isoformat(),
        'year': year,
        'total_261_records': len(df_261_raw),
        'total_101_records': len(df_101_raw),
        'total_historical_records': len(historical_data)
    }

    # Save to models/YEAR/ directory
    year_dir = os.path.join(models_dir, year)
    os.makedirs(year_dir, exist_ok=True)

    output_path = os.path.join(year_dir, "historical_data_precomputed.joblib")

    print(f"\n[7] Saving to {output_path}...")
    joblib.dump(precomputed, output_path, compress=3)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    File size: {file_size:.2f} MB")

    return True


def validate_deployment(models_dir: str):
    """
    Validate that deployment files are ready.
    """
    print(f"\n{'='*70}")
    print("VALIDATING DEPLOYMENT READINESS")
    print(f"{'='*70}")

    issues = []

    # Check for pre-computed files
    for year in ['2024', '2025']:
        precomputed_path = os.path.join(models_dir, year, "historical_data_precomputed.joblib")
        if os.path.exists(precomputed_path):
            size = os.path.getsize(precomputed_path) / (1024 * 1024)
            print(f"  [OK] {year}/historical_data_precomputed.joblib ({size:.2f} MB)")

            # Validate contents
            try:
                data = joblib.load(precomputed_path)
                if 'material_stats' not in data:
                    issues.append(f"{year}: missing material_stats")
                if 'kd_lookup' not in data:
                    issues.append(f"{year}: missing kd_lookup")
                if 'grade_breakdown_data' not in data:
                    issues.append(f"{year}: missing grade_breakdown_data (run prepare_deployment.py to update)")
            except Exception as e:
                issues.append(f"{year}: could not load file - {e}")
        else:
            issues.append(f"{year}: historical_data_precomputed.joblib not found")

    # Check for model files
    for year in ['2024', '2025']:
        model_path = os.path.join(models_dir, year, "yield_model.joblib")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  [OK] {year}/yield_model.joblib ({size:.2f} MB)")
        else:
            issues.append(f"{year}: yield_model.joblib not found")

    # Check dropdown options
    config_path = os.path.join(PROJECT_ROOT, "config", "dropdown_options.json")
    if os.path.exists(config_path):
        size = os.path.getsize(config_path) / 1024
        print(f"  [OK] config/dropdown_options.json ({size:.1f} KB)")
    else:
        issues.append("config/dropdown_options.json not found")

    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nDEPLOYMENT READY!")
        return True


def main():
    """Main entry point."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    data_dir = os.path.join(PROJECT_ROOT, "data")

    print("="*70)
    print("BTP CLOUD FOUNDRY DEPLOYMENT PREPARATION")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models directory: {models_dir}")
    print(f"Data directory: {data_dir}")

    # Get available years
    available_years = get_available_years()
    print(f"\nAvailable years with CSV data: {available_years}")

    if not available_years:
        print("ERROR: No CSV data files found. Please ensure 261_YYYY.csv and 101_YYYY.csv exist in data/")
        sys.exit(1)

    # Process each year
    success_count = 0
    for year in available_years:
        if prepare_deployment_for_year(year, models_dir, data_dir):
            success_count += 1

    print(f"\n\nProcessed {success_count}/{len(available_years)} years successfully")

    # Validate
    if validate_deployment(models_dir):
        print("\nYou can now deploy with: cf push")
        sys.exit(0)
    else:
        print("\nPlease resolve the issues above before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
