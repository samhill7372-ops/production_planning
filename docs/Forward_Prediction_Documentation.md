# Forward Prediction - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Input Requirements](#input-requirements)
3. [Technical Workflow](#technical-workflow)
4. [Historical Data Lookup](#historical-data-lookup)
5. [ML Model Prediction](#ml-model-prediction)
6. [Blending Logic](#blending-logic)
7. [Confidence Scoring System](#confidence-scoring-system)
8. [Yield Range Calculation](#yield-range-calculation)
9. [Output Calculations](#output-calculations)
10. [Recommendation Engine](#recommendation-engine)
11. [Multi-Input Support](#multi-input-support)
12. [Flow Diagrams](#flow-diagrams)

---

## Overview

### Purpose
Forward Prediction is a production planning feature that predicts **output materials and yields** based on input materials. It helps production planners answer the question: *"If I process this input material, what output can I expect and how confident can I be in that prediction?"*

### Key Benefits
- **Data-Driven Decisions**: Combines historical production data with machine learning predictions
- **Risk Assessment**: Provides confidence scores and risk levels for each prediction
- **Range Planning**: Offers min/max yield ranges for conservative and optimistic planning
- **Multi-Material Support**: Handles multiple input materials simultaneously
- **Recommendation Ranking**: Automatically ranks output options by likelihood of success

---

## Input Requirements

### Required Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| **Plant** | Manufacturing facility code | 1M02, 1Y01 |
| **Input Material** | Material code being processed | 4PO3BKS, 4PO2CKS |
| **Thickness** | Material thickness | 4, 6, 8 |
| **Species** | Wood species code | SM (Sugar Maple), AS (Ash), WO (White Oak) |
| **Grade** | Material grade classification | 2C, 1C, 3A, 3B |
| **Quantity (BFIN)** | Input quantity in Board Feet | 5000, 10000 |

### Optional Fields

| Field | Description | Default Behavior |
|-------|-------------|------------------|
| **Length** | Material length | Uses historical average if not specified |
| **Width** | Material width | Uses historical average if not specified |

### Multi-Input Mode
Users can add **multiple input materials** in a single prediction. The system will:
- Sum total input quantities
- Find all possible outputs from any input
- Average model predictions across inputs
- Mark results as "Multi-Input" model type

---

## Technical Workflow

### High-Level Process

```
Step 1: Collect User Inputs
           │
           ▼
Step 2: Search Historical Database
           │
           ▼
Step 3: Run ML Model Prediction
           │
           ▼
Step 4: Blend Historical + Model Results
           │
           ▼
Step 5: Calculate Confidence Scores
           │
           ▼
Step 6: Compute Yield Ranges (95% CI)
           │
           ▼
Step 7: Calculate Output BF Values
           │
           ▼
Step 8: Rank by Recommendation Score
           │
           ▼
Step 9: Return Results
```

### Code Location
- Main function: `simulate_output_materials_enhanced()` in `src/prediction_utils.py` (lines 676-858)
- UI handling: `app.py` (lines 257-389, 1799)

---

## Historical Data Lookup

### Process
1. Filter historical records matching the input material(s)
2. Group by unique output materials
3. Calculate statistics for each output:

### Statistics Collected

| Metric | Description | Formula |
|--------|-------------|---------|
| **Historical Yield** | Average yield percentage | `mean(Yield_Pct)` |
| **Yield Std Dev** | Consistency measure | `std(Yield_Pct)` |
| **Order Count** | Number of historical orders | `count(orders)` |
| **Total Input BF** | Historical input volume | `sum(Input_BF)` |
| **Total Output BF** | Historical output volume | `sum(Output_BF)` |

### Example
If input material "4PO3BKS" was processed 85 times historically:
- Average yield: 72.5%
- Standard deviation: 8.2%
- This data forms the "historical baseline" for predictions

---

## ML Model Prediction

### Feature Engineering
The ML model uses encoded features:

| Feature | Type | Encoding |
|---------|------|----------|
| Input_Material | Categorical | LabelEncoder |
| Input_Species | Categorical | LabelEncoder |
| Input_Grade | Categorical | LabelEncoder |
| Input_Plant | Categorical | LabelEncoder |
| Input_Thickness | Numeric | Raw value |
| Input_Length | Numeric | Raw value |
| Input_Width | Numeric | Raw value |
| Total_Input_BF | Numeric | Raw value |

### Model Types
- **2024 Model**: Ridge Regression (R² = 0.319)
- **2025 Model**: Gradient Boosting (R² = 0.287, RMSE = 3.94)

### Prediction Process
```python
1. Encode categorical inputs using trained LabelEncoders
2. Create feature vector in correct column order
3. Run model.predict(features)
4. Clip result to valid range [0%, 150%]
```

### Output
- Single yield percentage prediction per input-output combination

---

## Blending Logic

### The Weighted Average Formula

```
Weight = min(Historical_Order_Count / 100, 0.7)

Final_Yield = (Weight × Historical_Yield) + ((1 - Weight) × Model_Yield)
```

### Weight Distribution Table

| Historical Orders | Historical Weight | Model Weight | Reasoning |
|-------------------|-------------------|--------------|-----------|
| 1-9 orders | 1-9% | 91-99% | Very limited history, trust model |
| 10 orders | 10% | 90% | Some history, still favor model |
| 25 orders | 25% | 75% | Growing confidence in history |
| 50 orders | 50% | 50% | Equal trust |
| 75 orders | 70% | 30% | Strong historical evidence |
| 100+ orders | 70% (max) | 30% (min) | Maximum historical trust |

### Design Rationale
- **Minimum 30% model weight**: Ensures ML insights always contribute
- **Maximum 70% historical weight**: Prevents over-reliance on potentially outdated patterns
- **Linear scaling**: Smooth transition as data accumulates

---

## Confidence Scoring System

### Scoring Components (Total: 100 points)

#### 1. Historical Data Quantity (0-40 points)

| Order Count | Points | Interpretation |
|-------------|--------|----------------|
| 100+ orders | 40 | Extensive history |
| 50-99 orders | 30 | Good history |
| 20-49 orders | 20 | Moderate history |
| 5-19 orders | 10 | Limited history |
| < 5 orders | 5 | Minimal history |

#### 2. Yield Consistency (0-35 points)

| Std Deviation | Points | Interpretation |
|---------------|--------|----------------|
| ≤ 5% | 35 | Very consistent |
| 5-10% | 25 | Consistent |
| 10-15% | 15 | Moderate variation |
| 15-20% | 10 | High variation |
| > 20% | 5 | Very inconsistent |

#### 3. Model Performance (0-25 points)

| Model R² | Points | Interpretation |
|----------|--------|----------------|
| ≥ 0.70 | 25 | Excellent model |
| 0.50-0.69 | 20 | Good model |
| 0.30-0.49 | 15 | Moderate model |
| 0.10-0.29 | 10 | Weak model |
| < 0.10 | 5 | Poor model |

### Confidence Level Mapping

| Score Range | Confidence Level | Risk Level | Recommendation |
|-------------|------------------|------------|----------------|
| 80-100 | HIGH | LOW | Proceed with confidence |
| 60-79 | MEDIUM | MEDIUM | Proceed with monitoring |
| 40-59 | LOW | HIGH | Proceed with caution |
| 0-39 | VERY LOW | VERY HIGH | Requires manual review |

---

## Yield Range Calculation

### Statistical Basis
Uses 95% confidence interval based on historical yield distribution.

### Formulas

```
Standard Error (SE) = Historical_Std_Dev / √(Historical_Count)

Margin of Error (ME) = Z-score × SE
                     = 1.96 × SE  (for 95% confidence)

Yield_Min = max(0, Final_Yield - Margin_of_Error)
Yield_Max = min(150, Final_Yield + Margin_of_Error)

Yield_Range_Width = Yield_Max - Yield_Min
```

### Example Calculation

```
Given:
- Final Yield = 72.5%
- Historical Std Dev = 8.2%
- Historical Count = 85 orders

Calculation:
- SE = 8.2 / √85 = 8.2 / 9.22 = 0.89%
- ME = 1.96 × 0.89 = 1.74%
- Yield_Min = 72.5 - 1.74 = 70.76%
- Yield_Max = 72.5 + 1.74 = 74.24%

Result: Expected yield 72.5% with range [70.8%, 74.2%]
```

### Interpretation
- **Narrow range**: High confidence, consistent historical performance
- **Wide range**: Lower confidence, variable historical performance

---

## Output Calculations

### Board Feet Calculations

```
Output_BF_Expected = Input_BF × (Final_Yield / 100)
Output_BF_Min = Input_BF × (Yield_Min / 100)
Output_BF_Max = Input_BF × (Yield_Max / 100)
```

### Example

```
Given:
- Input BF = 10,000
- Final Yield = 72.5%
- Yield Range = [70.8%, 74.2%]

Results:
- Expected Output = 10,000 × 0.725 = 7,250 BF
- Minimum Output = 10,000 × 0.708 = 7,080 BF
- Maximum Output = 10,000 × 0.742 = 7,420 BF
```

---

## Recommendation Engine

### Recommendation Score Formula

```
Recommendation_Score = (Final_Yield / 100) × (Confidence_Score / 100)
```

### Score Interpretation

| Yield | Confidence | Recommendation Score | Ranking |
|-------|------------|---------------------|---------|
| 80% | 85 (HIGH) | 0.80 × 0.85 = 0.68 | Top choice |
| 90% | 50 (LOW) | 0.90 × 0.50 = 0.45 | Lower ranked |
| 70% | 70 (MEDIUM) | 0.70 × 0.70 = 0.49 | Middle ranked |

### Recommendation Strength

| Criteria | Strength |
|----------|----------|
| HIGH confidence AND yield ≥ 70% | STRONG |
| HIGH/MEDIUM confidence AND yield ≥ 60% | MODERATE |
| All other combinations | WEAK |

---

## Multi-Input Support

### How It Works
When multiple input materials are provided:

1. **Aggregate Inputs**: Sum total input BF across all materials
2. **Union Outputs**: Find all possible outputs from ANY input material
3. **Average Predictions**: Model yields are averaged across inputs
4. **Mark Source**: Results tagged as "Multi-Input" model type

### Benefits
- Broader output options discovered
- Combined production planning
- Realistic multi-material scenarios

### Example
```
Input 1: 5,000 BF of Material A → Possible outputs: X, Y, Z
Input 2: 5,000 BF of Material B → Possible outputs: Y, Z, W

Combined: 10,000 BF total → Possible outputs: X, Y, Z, W
```

---

## Flow Diagrams

### Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Plant  │ │Material │ │Thickness│ │ Species │ │  Grade  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       └──────────┬┴──────────┬┴──────────┬┴──────────┘          │
│                  │           │           │                      │
│              ┌───▼───────────▼───────────▼───┐                  │
│              │      Input BF Quantity        │                  │
│              └───────────────┬───────────────┘                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREDICTION ENGINE                            │
│                                                                  │
│  ┌────────────────────┐      ┌────────────────────┐             │
│  │  HISTORICAL DATA   │      │     ML MODEL       │             │
│  │                    │      │                    │             │
│  │ • Lookup by input  │      │ • Encode features  │             │
│  │ • Calc avg yield   │      │ • Run prediction   │             │
│  │ • Calc std dev     │      │ • Clip to [0,150]  │             │
│  │ • Count orders     │      │                    │             │
│  └─────────┬──────────┘      └─────────┬──────────┘             │
│            │                           │                         │
│            └─────────────┬─────────────┘                         │
│                          ▼                                       │
│            ┌─────────────────────────┐                          │
│            │    WEIGHTED BLENDING    │                          │
│            │                         │                          │
│            │ Weight = min(count/100, │                          │
│            │                   0.7)  │                          │
│            │                         │                          │
│            │ Final = W×Hist +        │                          │
│            │         (1-W)×Model     │                          │
│            └────────────┬────────────┘                          │
│                         │                                        │
│       ┌─────────────────┼─────────────────┐                     │
│       ▼                 ▼                 ▼                     │
│ ┌───────────┐    ┌───────────┐    ┌───────────────┐            │
│ │CONFIDENCE │    │  YIELD    │    │    OUTPUT     │            │
│ │  SCORING  │    │  RANGE    │    │  CALCULATION  │            │
│ │           │    │           │    │               │            │
│ │ • History │    │ • Std Err │    │ • Expected BF │            │
│ │   (40pts) │    │ • 95% CI  │    │ • Min BF      │            │
│ │ • Consist │    │ • Min/Max │    │ • Max BF      │            │
│ │   (35pts) │    │           │    │               │            │
│ │ • Model   │    │           │    │               │            │
│ │   (25pts) │    │           │    │               │            │
│ └─────┬─────┘    └─────┬─────┘    └───────┬───────┘            │
│       │                │                  │                     │
│       └────────────────┼──────────────────┘                     │
│                        ▼                                        │
│            ┌─────────────────────────┐                          │
│            │  RECOMMENDATION SCORE   │                          │
│            │                         │                          │
│            │ Score = Yield × Conf    │                          │
│            │ Sort by score DESC      │                          │
│            └────────────┬────────────┘                          │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                        OUTPUT DISPLAY                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Output Material │ Yield  │ Range      │ Confidence │ Risk  │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ 4PO3BKD         │ 72.5%  │ 70.8-74.2% │ HIGH (82)  │ LOW   │ │
│  │ 4PO2CKD         │ 68.2%  │ 65.1-71.3% │ MEDIUM(65) │ MED   │ │
│  │ 4PO3AKD         │ 61.0%  │ 55.2-66.8% │ LOW (45)   │ HIGH  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Top Recommendation: 4PO3BKD                                    │
│  Expected Output: 7,250 BF (Range: 7,080 - 7,420 BF)            │
│  Recommendation Strength: STRONG                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Summary

Forward Prediction provides a robust, data-driven approach to production planning by:

1. **Combining data sources**: Historical patterns + ML predictions
2. **Quantifying uncertainty**: Confidence scores + yield ranges
3. **Enabling planning**: Min/max output ranges for conservative/optimistic scenarios
4. **Ranking options**: Automatic recommendation scoring

This approach ensures planners have both the predicted values AND the context needed to make informed decisions.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Source Code: `src/prediction_utils.py`, `app.py`*
