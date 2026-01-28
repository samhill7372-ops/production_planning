"""
BF Calculation Demo - Sample File

This script demonstrates the calculation logic for:
- Current method: Count-based Historical %
- New method: BF Output-based Historical %

Run: python samples/bf_calculation_demo.py
"""

import pandas as pd


def create_mock_data():
    """
    Create mock historical data based on client's KS example:
    - Input Material: KS
    - Total Input BF (Consumed): 100,000
    - Total Output BF (Produced): 91,000

    Output Materials:
    - KS1C: 10% of produced
    - KS2C: 20% of produced
    - KS3C: 40% of produced
    - KS4C: 30% of produced (remaining)
    """

    # Mock historical data - simulating multiple orders
    # Each row represents a manufacturing order
    historical_data = pd.DataFrame({
        'MANUFACTURINGORDER': [f'MO{i:04d}' for i in range(1, 101)],  # 100 orders
        'Input_Material': ['KS'] * 100,
        'Total_Input_BF': [1000] * 100,  # Each order: 1000 BF input = 100,000 total
    })

    # Output data - multiple outputs per order
    # Distributing to match the percentages
    output_records = []

    for i in range(1, 101):
        order_id = f'MO{i:04d}'

        # Distribute 910 BF output per order (91% yield)
        # KS1C: 10% = 91 BF per order
        # KS2C: 20% = 182 BF per order
        # KS3C: 40% = 364 BF per order
        # KS4C: 30% = 273 BF per order

        outputs = [
            {'MANUFACTURINGORDER': order_id, 'Output_Material': 'KS1C', 'Total_Output_BF': 91},
            {'MANUFACTURINGORDER': order_id, 'Output_Material': 'KS2C', 'Total_Output_BF': 182},
            {'MANUFACTURINGORDER': order_id, 'Output_Material': 'KS3C', 'Total_Output_BF': 364},
            {'MANUFACTURINGORDER': order_id, 'Output_Material': 'KS4C', 'Total_Output_BF': 273},
        ]
        output_records.extend(outputs)

    output_data = pd.DataFrame(output_records)

    return historical_data, output_data


def calculate_count_based(output_data):
    """
    CURRENT METHOD: Count-based calculation
    Historical % = (Order_Count * 100) / Total_Orders
    """
    # Count orders per output material
    count_summary = output_data.groupby('Output_Material').agg(
        Order_Count=('MANUFACTURINGORDER', 'nunique')
    ).reset_index()

    total_orders = count_summary['Order_Count'].sum()
    count_summary['Historical_Pct'] = (count_summary['Order_Count'] * 100) / total_orders

    return count_summary, total_orders


def calculate_bf_based(output_data):
    """
    NEW METHOD: BF Output-based calculation
    Historical % = (Material_BF_Output * 100) / Total_BF_Output
    """
    # Sum BF output per output material
    bf_summary = output_data.groupby('Output_Material').agg(
        Total_BF_Output=('Total_Output_BF', 'sum')
    ).reset_index()

    total_bf_output = bf_summary['Total_BF_Output'].sum()
    bf_summary['Historical_Pct'] = (bf_summary['Total_BF_Output'] * 100) / total_bf_output

    return bf_summary, total_bf_output


def calculate_expected_bf(input_bf, total_input_bf, total_output_bf, material_pct):
    """
    Calculate Expected BF for a new input.

    Expected Output = New Input BF * (Historical Output / Historical Input)
    Expected Material BF = Expected Output * (Material % / 100)
    """
    # Calculate yield ratio
    yield_ratio = total_output_bf / total_input_bf

    # Expected total output
    expected_output = input_bf * yield_ratio

    # Expected BF for this material
    expected_material_bf = expected_output * (material_pct / 100)

    return expected_output, expected_material_bf


def main():
    print("=" * 70)
    print("BF CALCULATION DEMO")
    print("=" * 70)

    # Create mock data
    historical_input, historical_output = create_mock_data()

    # Calculate totals
    total_input_bf = historical_input['Total_Input_BF'].sum()
    total_output_bf = historical_output['Total_Output_BF'].sum()

    print("\n" + "-" * 70)
    print("HISTORICAL DATA SUMMARY (Input Material: KS)")
    print("-" * 70)
    print(f"Total Input BF (Consumed):   {total_input_bf:>12,}")
    print(f"Total Output BF (Produced):  {total_output_bf:>12,}")
    print(f"Overall Yield:               {(total_output_bf/total_input_bf)*100:>11.1f}%")

    # Method 1: Count-based (CURRENT)
    print("\n" + "-" * 70)
    print("METHOD 1: COUNT-BASED (CURRENT)")
    print("-" * 70)
    count_summary, total_orders = calculate_count_based(historical_output)
    print(f"Total Orders: {total_orders}")
    print()
    print(f"{'Material':<12} {'Count':>10} {'Historical %':>15}")
    print("-" * 40)
    for _, row in count_summary.iterrows():
        print(f"{row['Output_Material']:<12} {row['Order_Count']:>10} {row['Historical_Pct']:>14.1f}%")

    # Method 2: BF-based (NEW)
    print("\n" + "-" * 70)
    print("METHOD 2: BF OUTPUT-BASED (NEW)")
    print("-" * 70)
    bf_summary, total_bf = calculate_bf_based(historical_output)
    print(f"Total BF Output: {total_bf:,}")
    print()
    print(f"{'Material':<12} {'BF Output':>12} {'Historical %':>15}")
    print("-" * 45)
    for _, row in bf_summary.iterrows():
        print(f"{row['Output_Material']:<12} {row['Total_BF_Output']:>12,} {row['Historical_Pct']:>14.1f}%")

    # Forward Prediction Example
    print("\n" + "-" * 70)
    print("FORWARD PREDICTION EXAMPLE")
    print("-" * 70)
    new_input_bf = 90000
    print(f"New Input BF: {new_input_bf:,}")
    print()

    expected_total_output, _ = calculate_expected_bf(
        new_input_bf, total_input_bf, total_output_bf, 100
    )
    print(f"Expected Total Output: {expected_total_output:,.0f} BF")
    print()

    print(f"{'Material':<12} {'Historical %':>15} {'Expected BF':>15}")
    print("-" * 45)
    for _, row in bf_summary.iterrows():
        _, expected_bf = calculate_expected_bf(
            new_input_bf, total_input_bf, total_output_bf, row['Historical_Pct']
        )
        print(f"{row['Output_Material']:<12} {row['Historical_Pct']:>14.1f}% {expected_bf:>14,.0f}")

    # Comparison Table
    print("\n" + "-" * 70)
    print("COMPARISON: COUNT vs BF OUTPUT")
    print("-" * 70)

    comparison = count_summary.merge(bf_summary, on='Output_Material', suffixes=('_count', '_bf'))
    print(f"{'Material':<12} {'Count':>8} {'Count %':>10} {'BF Output':>12} {'BF %':>10}")
    print("-" * 55)
    for _, row in comparison.iterrows():
        print(f"{row['Output_Material']:<12} {row['Order_Count']:>8} {row['Historical_Pct_count']:>9.1f}% {row['Total_BF_Output']:>12,} {row['Historical_Pct_bf']:>9.1f}%")

    print("\n" + "=" * 70)
    print("KEY DIFFERENCE:")
    print("- Count method: All materials have equal % (same # of orders)")
    print("- BF method: % reflects actual production volume")
    print("=" * 70)


if __name__ == "__main__":
    main()
