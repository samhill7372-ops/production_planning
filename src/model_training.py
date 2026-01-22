"""
Model Training Module for Material Yield Prediction System

SAP Manufacturing Logic:
- Movement Type 261 = Goods Issue to Order = INPUT (raw materials CONSUMED from stock)
- Movement Type 101 = Goods Receipt = OUTPUT (finished goods RECEIVED into stock)
- Yield = Total_Output_BF (from 101) / Total_Input_BF (from 261)
- Trains regression models to predict Yield_Percentage
- Supports single-input and multi-input material scenarios
- Output material simulation based on historical patterns

Real-World Production Planning Use Cases:
1. Forward Planning: "If I consume X BF of raw material, how much output will I get?"
2. Reverse Planning: "If I need Y BF of finished goods, how much raw material do I need?"
3. Material Selection: "Which raw material gives the best yield for my needs?"
4. Anomaly Detection: "Is this manufacturing order producing abnormal loss?"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any, Optional, List, Union
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class YieldPrediction:
    """Structured yield prediction result with confidence bounds."""
    predicted_yield_pct: float
    lower_bound_pct: float
    upper_bound_pct: float
    confidence_level: float
    model_used: str

    def expected_output_bf(self, input_bf: float) -> float:
        """Calculate expected output board feet."""
        return input_bf * self.predicted_yield_pct / 100

    def output_range_bf(self, input_bf: float) -> Tuple[float, float]:
        """Calculate output range (lower, upper) in board feet."""
        return (
            input_bf * self.lower_bound_pct / 100,
            input_bf * self.upper_bound_pct / 100
        )


@dataclass
class MaterialRequirement:
    """Result of reverse planning calculation."""
    required_input_bf: float
    target_output_bf: float
    predicted_yield_pct: float
    conservative_input_bf: float  # Based on lower yield bound
    optimistic_input_bf: float    # Based on upper yield bound
    safety_margin_pct: float


class YieldPredictionModel:
    """
    Comprehensive Yield Prediction Model supporting:
    - Single input material prediction
    - Multiple input materials aggregation
    - Output material yield ranking
    - Model comparison and selection

    Real-World Use Cases:
    - Forward Planning: predict_forward() - Input BF → Expected Output BF
    - Reverse Planning: predict_reverse() - Required Output BF → Required Input BF
    - Material Comparison: compare_materials() - Rank materials by effective yield
    - Anomaly Detection: detect_anomaly() - Flag abnormal yield orders
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.encoders = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.trained = False
        self.historical_data = None
        self.test_results = None
        # Store yield statistics per material for confidence intervals
        self.yield_stats_by_material = {}
        # Cross-validation results
        self.cv_results = {}

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Prepare feature matrix with available columns."""
        available = [c for c in feature_columns if c in df.columns]
        X = df[available].copy()

        # Fill missing values
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(0, inplace=True)

        return X

    def _compute_yield_stats_by_material(self, df: pd.DataFrame) -> None:
        """Pre-compute yield statistics per input material for confidence intervals."""
        if 'Input_Material' not in df.columns or 'Yield_Percentage' not in df.columns:
            return

        for material in df['Input_Material'].unique():
            subset = df[df['Input_Material'] == material]['Yield_Percentage']
            if len(subset) >= 3:
                self.yield_stats_by_material[material] = {
                    'mean': subset.mean(),
                    'std': subset.std(),
                    'count': len(subset),
                    'q25': subset.quantile(0.25),
                    'q75': subset.quantile(0.75),
                    'min': subset.min(),
                    'max': subset.max()
                }

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'Yield_Percentage',
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train multiple regression models for yield prediction.

        Features used:
        - Input_Material_Encoded, Input_Specie_Encoded, Input_Grade_Encoded
        - Input_Plant_Encoded, Input_Thickness, Input_Length, Input_Width
        - Total_Input_BF

        Target: Yield_Percentage

        Why Yield % is the correct target:
        - Normalized metric independent of order size
        - Captures material efficiency and process losses
        - Enables both forward and reverse planning
        """
        print("=" * 70)
        print("MODEL TRAINING (Yield Percentage Prediction)")
        print("=" * 70)

        # Store historical data for output simulation
        self.historical_data = df.copy()
        self.feature_columns = [c for c in feature_columns if c in df.columns]

        # Pre-compute yield statistics for confidence intervals
        self._compute_yield_stats_by_material(df)
        print(f"Computed yield statistics for {len(self.yield_stats_by_material)} materials")

        # Prepare features
        X = self.prepare_features(df, self.feature_columns)
        y = df[target_column].copy()

        print(f"\nFeatures used: {list(X.columns)}")
        print(f"Target: {target_column}")
        print(f"Dataset: {len(X):,} samples")
        print(f"Yield range: {y.min():.1f}% - {y.max():.1f}% (mean: {y.mean():.1f}%)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Scale features for linear models
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)

        # Define models to train
        # Why Gradient Boosted Trees (XGBoost/GradientBoosting) are appropriate:
        # 1. Handle mixed feature types (categorical encoded + numeric) natively
        # 2. Capture non-linear relationships (e.g., thickness × grade interactions)
        # 3. Robust to outliers in manufacturing data
        # 4. Provide feature importance for interpretability
        # 5. Fast inference for production deployment
        model_configs = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=random_state
            ),
            'Ridge': Ridge(alpha=1.0)
        }

        # Add XGBoost if available - optimized for manufacturing yield prediction
        if HAS_XGBOOST:
            model_configs['XGBoost'] = XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )

        results = {}
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Train each model
        for name, model in model_configs.items():
            print(f"\nTraining {name}...")

            use_scaled = name in ['Ridge', 'LinearRegression']

            # Cross-validation for robust evaluation
            if use_scaled:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            # Store cross-validation results
            self.cv_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }

            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # MAPE (avoid division by zero)
            mask = y_test > 0
            if mask.sum() > 0:
                test_mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
            else:
                test_mape = np.nan

            self.models[name] = model
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'MAE': test_mae,
                'RMSE': test_rmse,
                'MAPE': test_mape,
                'R2': test_r2,
                'y_pred': y_pred_test
            }

            print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  MAE: {test_mae:.2f}% | RMSE: {test_rmse:.2f}%")

        # Select best model
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = self.models[self.best_model_name]

        print(f"\n{'=' * 70}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test R² = {results[self.best_model_name]['test_r2']:.4f}")
        print(f"{'=' * 70}")

        # Store test results
        self.test_results = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': results[self.best_model_name]['y_pred'],
            'Error': y_test.values - results[self.best_model_name]['y_pred']
        })
        self.test_results['Error_Pct'] = (
            self.test_results['Error'] / self.test_results['Actual']
        ) * 100

        self.metrics = results
        self.trained = True

        return results

    def predict_yield(
        self,
        input_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> float:
        """Predict yield for a single input configuration."""
        if not self.trained:
            raise ValueError("Model not trained")

        model = self.models.get(model_name, self.best_model)
        use_scaled = (model_name or self.best_model_name) in ['Ridge', 'LinearRegression']

        # Build feature vector
        X = pd.DataFrame([input_data])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns].fillna(0)

        if use_scaled:
            X = self.scalers['standard'].transform(X)

        pred = model.predict(X)[0]
        return np.clip(pred, 0, 150)  # Reasonable yield bounds

    def predict_yield_with_confidence(
        self,
        input_data: Dict[str, Any],
        input_material: Optional[str] = None,
        confidence_level: float = 0.9
    ) -> YieldPrediction:
        """
        Predict yield with confidence intervals.

        Returns a YieldPrediction object with:
        - predicted_yield_pct: Point estimate
        - lower_bound_pct / upper_bound_pct: Confidence interval
        - confidence_level: The confidence level used

        Example:
            pred = model.predict_yield_with_confidence(input_data, "6SM2CKD")
            print(f"Expected yield: {pred.predicted_yield_pct:.1f}%")
            print(f"90% CI: [{pred.lower_bound_pct:.1f}%, {pred.upper_bound_pct:.1f}%]")
        """
        predicted = self.predict_yield(input_data)

        # Get material-specific statistics for confidence interval
        material = input_material or input_data.get('Input_Material', '')
        if material in self.yield_stats_by_material:
            stats = self.yield_stats_by_material[material]
            # Use historical std for confidence interval
            z_score = 1.645 if confidence_level == 0.9 else 1.96  # 90% or 95% CI
            margin = z_score * stats['std']
            lower = max(0, predicted - margin)
            upper = min(150, predicted + margin)
        else:
            # Fallback: use global model RMSE as uncertainty
            rmse = self.metrics.get(self.best_model_name, {}).get('RMSE', 5.0)
            z_score = 1.645 if confidence_level == 0.9 else 1.96
            lower = max(0, predicted - z_score * rmse)
            upper = min(150, predicted + z_score * rmse)

        return YieldPrediction(
            predicted_yield_pct=round(predicted, 2),
            lower_bound_pct=round(lower, 2),
            upper_bound_pct=round(upper, 2),
            confidence_level=confidence_level,
            model_used=self.best_model_name
        )

    # =========================================================================
    # REAL-WORLD PRODUCTION PLANNING METHODS
    # =========================================================================

    def predict_forward(
        self,
        input_bf: float,
        input_data: Dict[str, Any],
        input_material: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        FORWARD PLANNING: "If I consume X board feet of raw material,
        how much usable output will I get?"

        This is the fundamental question for production scheduling.

        Args:
            input_bf: Board feet of raw material to consume
            input_data: Feature dict with encoded material properties
            input_material: Optional material name for confidence intervals

        Returns:
            Dict with expected output, range, and confidence info

        Example:
            result = model.predict_forward(
                input_bf=10000,
                input_data={'Input_Material_Encoded': 5, ...},
                input_material='6SM2CKD'
            )
            print(f"Expected output: {result['expected_output_bf']:,.0f} BF")
            print(f"Range: {result['pessimistic_output_bf']:,.0f} - {result['optimistic_output_bf']:,.0f} BF")
        """
        pred = self.predict_yield_with_confidence(input_data, input_material)

        return {
            'input_bf': input_bf,
            'predicted_yield_pct': pred.predicted_yield_pct,
            'expected_output_bf': round(input_bf * pred.predicted_yield_pct / 100, 2),
            'pessimistic_output_bf': round(input_bf * pred.lower_bound_pct / 100, 2),
            'optimistic_output_bf': round(input_bf * pred.upper_bound_pct / 100, 2),
            'yield_lower_bound_pct': pred.lower_bound_pct,
            'yield_upper_bound_pct': pred.upper_bound_pct,
            'confidence_level': pred.confidence_level,
            'model_used': pred.model_used
        }

    def predict_reverse(
        self,
        required_output_bf: float,
        input_data: Dict[str, Any],
        input_material: Optional[str] = None,
        safety_margin_pct: float = 5.0
    ) -> MaterialRequirement:
        """
        REVERSE PLANNING: "If I need Y board feet of finished goods,
        how much raw material do I need to procure?"

        This is critical for procurement and inventory planning.

        Args:
            required_output_bf: Target output in board feet
            input_data: Feature dict with encoded material properties
            input_material: Optional material name for confidence intervals
            safety_margin_pct: Additional safety buffer (default 5%)

        Returns:
            MaterialRequirement with required input quantities

        Example:
            req = model.predict_reverse(
                required_output_bf=5000,
                input_data={'Input_Material_Encoded': 5, ...},
                input_material='6SM2CKD',
                safety_margin_pct=5.0
            )
            print(f"Need to order: {req.required_input_bf:,.0f} BF")
            print(f"Conservative estimate: {req.conservative_input_bf:,.0f} BF")
        """
        pred = self.predict_yield_with_confidence(input_data, input_material)

        # Basic calculation
        if pred.predicted_yield_pct > 0:
            required_input = required_output_bf / (pred.predicted_yield_pct / 100)
        else:
            required_input = required_output_bf  # Fallback

        # Conservative (use lower yield bound)
        if pred.lower_bound_pct > 0:
            conservative_input = required_output_bf / (pred.lower_bound_pct / 100)
        else:
            conservative_input = required_input * 1.2

        # Optimistic (use upper yield bound)
        if pred.upper_bound_pct > 0:
            optimistic_input = required_output_bf / (pred.upper_bound_pct / 100)
        else:
            optimistic_input = required_input * 0.9

        # Add safety margin to required input
        required_with_safety = required_input * (1 + safety_margin_pct / 100)

        return MaterialRequirement(
            required_input_bf=round(required_with_safety, 2),
            target_output_bf=required_output_bf,
            predicted_yield_pct=pred.predicted_yield_pct,
            conservative_input_bf=round(conservative_input, 2),
            optimistic_input_bf=round(optimistic_input, 2),
            safety_margin_pct=safety_margin_pct
        )

    def compare_materials(
        self,
        material_options: List[Dict[str, Any]],
        price_per_bf: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        MATERIAL SELECTION: "Among multiple raw materials,
        which one gives the best yield / lowest effective cost?"

        Critical for sourcing decisions.

        Args:
            material_options: List of dicts with material properties
                Each dict should have 'Input_Material' and encoded features
            price_per_bf: Optional dict mapping material name to price per BF

        Returns:
            DataFrame comparing materials by yield and effective cost

        Example:
            options = [
                {'Input_Material': 'MAT_A', 'Input_Material_Encoded': 1, ...},
                {'Input_Material': 'MAT_B', 'Input_Material_Encoded': 2, ...},
            ]
            prices = {'MAT_A': 2.10, 'MAT_B': 2.40}
            comparison = model.compare_materials(options, prices)
        """
        results = []

        for opt in material_options:
            material_name = opt.get('Input_Material', 'Unknown')
            pred = self.predict_yield_with_confidence(opt, material_name)

            row = {
                'Material': material_name,
                'Predicted_Yield_Pct': pred.predicted_yield_pct,
                'Yield_Lower_Pct': pred.lower_bound_pct,
                'Yield_Upper_Pct': pred.upper_bound_pct,
            }

            # Calculate effective cost if prices provided
            if price_per_bf and material_name in price_per_bf:
                price = price_per_bf[material_name]
                row['Price_Per_BF_Input'] = price
                # Effective cost = input price / yield
                if pred.predicted_yield_pct > 0:
                    row['Effective_Cost_Per_BF_Output'] = round(
                        price / (pred.predicted_yield_pct / 100), 4
                    )
                else:
                    row['Effective_Cost_Per_BF_Output'] = float('inf')

            # Add historical stats if available
            if material_name in self.yield_stats_by_material:
                stats = self.yield_stats_by_material[material_name]
                row['Historical_Orders'] = stats['count']
                row['Historical_Mean_Yield'] = round(stats['mean'], 2)

            results.append(row)

        df = pd.DataFrame(results)

        # Sort by effective cost if available, otherwise by yield
        if 'Effective_Cost_Per_BF_Output' in df.columns:
            df = df.sort_values('Effective_Cost_Per_BF_Output')
        else:
            df = df.sort_values('Predicted_Yield_Pct', ascending=False)

        return df

    def detect_anomaly(
        self,
        actual_yield_pct: float,
        input_data: Dict[str, Any],
        input_material: Optional[str] = None,
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        ANOMALY DETECTION: "Is this manufacturing order producing abnormal loss?"

        Flags orders where actual yield deviates significantly from expected.

        Args:
            actual_yield_pct: The actual observed yield percentage
            input_data: Feature dict with encoded material properties
            input_material: Optional material name for baseline
            threshold_std: Number of standard deviations to flag (default 2.0)

        Returns:
            Dict with anomaly status and details

        Example:
            result = model.detect_anomaly(
                actual_yield_pct=65.0,
                input_data={'Input_Material_Encoded': 5, ...},
                input_material='6SM2CKD',
                threshold_std=2.0
            )
            if result['is_anomaly']:
                print(f"ALERT: Order yield {result['deviation_std']:.1f} std below expected")
        """
        pred = self.predict_yield_with_confidence(input_data, input_material)
        predicted = pred.predicted_yield_pct

        # Get standard deviation for this material
        material = input_material or input_data.get('Input_Material', '')
        if material in self.yield_stats_by_material:
            std = self.yield_stats_by_material[material]['std']
        else:
            # Use model RMSE as proxy for std
            std = self.metrics.get(self.best_model_name, {}).get('RMSE', 5.0)

        # Calculate deviation
        deviation = actual_yield_pct - predicted
        deviation_std = deviation / std if std > 0 else 0

        is_anomaly = abs(deviation_std) > threshold_std
        anomaly_type = None
        if is_anomaly:
            anomaly_type = 'LOW_YIELD' if deviation < 0 else 'HIGH_YIELD'

        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'actual_yield_pct': actual_yield_pct,
            'predicted_yield_pct': predicted,
            'deviation_pct': round(deviation, 2),
            'deviation_std': round(deviation_std, 2),
            'threshold_std': threshold_std,
            'material': material,
            'possible_causes': self._get_anomaly_causes(anomaly_type, deviation_std)
        }

    def _get_anomaly_causes(
        self,
        anomaly_type: Optional[str],
        deviation_std: float
    ) -> List[str]:
        """Get possible causes for yield anomaly."""
        if anomaly_type is None:
            return []

        if anomaly_type == 'LOW_YIELD':
            causes = [
                'Equipment malfunction or misalignment',
                'Raw material quality issue (defects, knots)',
                'Operator error or training gap',
                'Recording error in input or output quantities',
                'Process parameter drift',
            ]
            if abs(deviation_std) > 3:
                causes.insert(0, 'CRITICAL: Severe process deviation - investigate immediately')
        else:  # HIGH_YIELD
            causes = [
                'Recording error (underreported input or overreported output)',
                'Inventory adjustment not captured',
                'Exceptionally good raw material quality',
                'Process improvement not yet reflected in baseline',
            ]

        return causes

    def simulate_output_materials(
        self,
        input_materials: List[Dict[str, Any]],
        encoders: Dict
    ) -> pd.DataFrame:
        """
        Given input materials, simulate possible output materials and their yields.

        SAP Logic: Input materials produce DIFFERENT output materials.
        Uses historical data to identify which outputs are possible.
        """
        if self.historical_data is None:
            return pd.DataFrame()

        results = []
        total_input_bf = sum(m.get('Total_Input_BF', 0) for m in input_materials)

        # For each input material, find historical output patterns
        for inp in input_materials:
            input_mat = inp.get('Input_Material', '')
            input_bf = inp.get('Total_Input_BF', 0)

            # Get historical records for this input
            if 'Input_Material' in self.historical_data.columns:
                hist = self.historical_data[
                    self.historical_data['Input_Material'] == input_mat
                ]
            else:
                hist = self.historical_data

            if len(hist) == 0:
                hist = self.historical_data  # Use all data if no match

            # Get unique output materials and their statistics
            if 'Output_Material' in hist.columns:
                for output_mat in hist['Output_Material'].unique():
                    output_hist = hist[hist['Output_Material'] == output_mat]

                    if len(output_hist) == 0:
                        continue

                    # Historical statistics
                    hist_yield = output_hist['Yield_Percentage'].mean()
                    hist_std = output_hist['Yield_Percentage'].std()
                    hist_count = len(output_hist)

                    # Model prediction
                    try:
                        pred_yield = self.predict_yield(inp)
                    except:
                        pred_yield = hist_yield

                    # Combine model and historical (weighted average)
                    # Weight historical more if we have more data points
                    weight = min(hist_count / 100, 0.7)  # Max 70% weight to historical
                    final_yield = weight * hist_yield + (1 - weight) * pred_yield

                    results.append({
                        'Input_Material': input_mat,
                        'Output_Material': output_mat,
                        'Predicted_Yield_Pct': round(final_yield, 2),
                        'Model_Yield_Pct': round(pred_yield, 2),
                        'Historical_Yield_Pct': round(hist_yield, 2),
                        'Yield_Std': round(hist_std, 2) if not np.isnan(hist_std) else 0,
                        'Historical_Orders': hist_count,
                        'Input_BF': input_bf,
                        'Predicted_Output_BF': round(input_bf * final_yield / 100, 2),
                        'Model_Type': 'Single-Input'
                    })

        if len(results) == 0:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        # Aggregate by output material
        agg_df = result_df.groupby('Output_Material').agg({
            'Predicted_Yield_Pct': 'mean',
            'Model_Yield_Pct': 'mean',
            'Historical_Yield_Pct': 'mean',
            'Yield_Std': 'mean',
            'Historical_Orders': 'sum',
            'Input_BF': 'sum',
            'Predicted_Output_BF': 'sum',
            'Model_Type': 'first'
        }).reset_index()

        # Add multi-input indicator if multiple inputs
        if len(input_materials) > 1:
            agg_df['Model_Type'] = 'Multi-Input'

        agg_df = agg_df.sort_values('Predicted_Yield_Pct', ascending=False)

        return agg_df

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        for name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                return pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': self.models[name].feature_importances_
                }).sort_values('Importance', ascending=False)
        return pd.DataFrame()

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison table of all trained models with cross-validation results."""
        rows = []
        for name, m in self.metrics.items():
            if isinstance(m, dict):
                row = {
                    'Model': name,
                    'Train R²': round(m.get('train_r2', 0), 4),
                    'Test R²': round(m.get('test_r2', 0), 4),
                    'CV R² (mean)': round(m.get('cv_r2_mean', 0), 4),
                    'CV R² (std)': round(m.get('cv_r2_std', 0), 4),
                    'MAE': round(m.get('MAE', 0), 2),
                    'RMSE': round(m.get('RMSE', 0), 2),
                    'MAPE': round(m.get('MAPE', 0), 2) if not np.isnan(m.get('MAPE', 0)) else 'N/A'
                }
                rows.append(row)
        return pd.DataFrame(rows).sort_values('Test R²', ascending=False)

    def save(self, path: str = "."):
        """Save all model artifacts."""
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.best_model, os.path.join(path, 'yield_model.joblib'))
        joblib.dump(self.models, os.path.join(path, 'all_models.joblib'))
        joblib.dump(self.scalers, os.path.join(path, 'scalers.joblib'))
        joblib.dump(self.feature_columns, os.path.join(path, 'feature_columns.joblib'))
        joblib.dump(self.metrics, os.path.join(path, 'metrics.joblib'))
        joblib.dump(self.encoders, os.path.join(path, 'encoders.joblib'))

        # Save new attributes for confidence intervals and CV
        joblib.dump(self.yield_stats_by_material, os.path.join(path, 'yield_stats.joblib'))
        joblib.dump(self.cv_results, os.path.join(path, 'cv_results.joblib'))

        if self.test_results is not None:
            self.test_results.to_csv(os.path.join(path, 'test_results.csv'), index=False)

        importance = self.get_feature_importance()
        if len(importance) > 0:
            importance.to_csv(os.path.join(path, 'feature_importance.csv'), index=False)

        print(f"Model artifacts saved to: {path}")

    def load(self, path: str = "."):
        """Load model artifacts."""
        self.best_model = joblib.load(os.path.join(path, 'yield_model.joblib'))

        if os.path.exists(os.path.join(path, 'all_models.joblib')):
            self.models = joblib.load(os.path.join(path, 'all_models.joblib'))

        if os.path.exists(os.path.join(path, 'scalers.joblib')):
            self.scalers = joblib.load(os.path.join(path, 'scalers.joblib'))

        self.feature_columns = joblib.load(os.path.join(path, 'feature_columns.joblib'))
        self.metrics = joblib.load(os.path.join(path, 'metrics.joblib'))

        if os.path.exists(os.path.join(path, 'encoders.joblib')):
            self.encoders = joblib.load(os.path.join(path, 'encoders.joblib'))

        if os.path.exists(os.path.join(path, 'test_results.csv')):
            self.test_results = pd.read_csv(os.path.join(path, 'test_results.csv'))

        # Load new attributes for confidence intervals and CV
        if os.path.exists(os.path.join(path, 'yield_stats.joblib')):
            self.yield_stats_by_material = joblib.load(os.path.join(path, 'yield_stats.joblib'))

        if os.path.exists(os.path.join(path, 'cv_results.joblib')):
            self.cv_results = joblib.load(os.path.join(path, 'cv_results.joblib'))

        # Find best model name from metrics
        if self.metrics:
            self.best_model_name = max(
                self.metrics.keys(),
                key=lambda k: self.metrics[k].get('test_r2', 0) if isinstance(self.metrics[k], dict) else 0
            )

        self.trained = True
        print(f"Model loaded from: {path}")


def train_yield_model(
    df: pd.DataFrame,
    encoders: Dict,
    save_path: str = None
) -> YieldPredictionModel:
    """
    Complete training pipeline.

    Trains models to predict Yield_Percentage based on input material features.
    """
    # Use default models directory if not provided
    if save_path is None:
        save_path = MODELS_DIR

    # Define feature columns
    feature_columns = [
        'Input_Material_Encoded',
        'Input_Specie_Encoded',
        'Input_Grade_Encoded',
        'Input_Plant_Encoded',
        'Input_Thickness',
        'Input_Length',
        'Input_Width',
        'Total_Input_BF'
    ]

    # Filter to available
    feature_columns = [c for c in feature_columns if c in df.columns]

    # Create and train model
    model = YieldPredictionModel()
    model.encoders = encoders
    model.train(df, feature_columns, target_column='Yield_Percentage')

    # Save
    model.save(save_path)

    return model


# =============================================================================
# OUTPUT MATERIAL CLASSIFICATION MODEL
# =============================================================================

class OutputMaterialClassifier:
    """
    Classification model to predict which Output Material will be produced
    given Input Material characteristics.

    This answers: "If I use this input material, what output will I get?"

    Uses multi-class classification since one input can potentially produce
    different outputs depending on other factors.
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = []
        self.label_encoder = None  # For Output_Material
        self.metrics = {}
        self.trained = False
        self.class_names = []

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'Output_Material_Encoded',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train classification models to predict Output Material.

        Features: Input material properties (same as yield model)
        Target: Output_Material_Encoded (categorical)
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

        print("=" * 70)
        print("OUTPUT MATERIAL CLASSIFICATION MODEL")
        print("=" * 70)

        self.feature_columns = [c for c in feature_columns if c in df.columns]

        # Prepare features
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()

        # Fill missing values
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(0, inplace=True)

        # Get class names for later use
        if 'Output_Material' in df.columns:
            self.class_names = df['Output_Material'].unique().tolist()

        n_classes = y.nunique()
        print(f"\nFeatures: {list(X.columns)}")
        print(f"Target: {target_column}")
        print(f"Dataset: {len(X):,} samples")
        print(f"Number of output material classes: {n_classes}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if n_classes < 50 else None
        )
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Define classification models
        model_configs = {
            'RandomForest_Classifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ),
            'GradientBoosting_Classifier': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=random_state
            )
        }

        # Add XGBoost classifier if available
        if HAS_XGBOOST:
            from xgboost import XGBClassifier
            model_configs['XGBoost_Classifier'] = XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

        results = {}

        for name, model in model_configs.items():
            print(f"\nTraining {name}...")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)

                # Top-3 accuracy (if we have probabilities and enough classes)
                top3_acc = None
                if y_pred_proba is not None and n_classes >= 3:
                    try:
                        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=min(3, n_classes))
                    except:
                        top3_acc = None

                self.models[name] = model
                results[name] = {
                    'accuracy': accuracy,
                    'top3_accuracy': top3_acc,
                    'n_classes': n_classes
                }

                print(f"  Accuracy: {accuracy:.4f}")
                if top3_acc:
                    print(f"  Top-3 Accuracy: {top3_acc:.4f}")

            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue

        # Select best model
        if results:
            self.best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            self.best_model = self.models[self.best_model_name]
            self.metrics = results
            self.trained = True

            print(f"\n{'=' * 70}")
            print(f"BEST CLASSIFIER: {self.best_model_name}")
            print(f"Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
            print(f"{'=' * 70}")

        return results

    def predict(self, input_data: Dict[str, Any]) -> int:
        """Predict the output material class (encoded)."""
        if not self.trained:
            raise ValueError("Model not trained")

        X = pd.DataFrame([input_data])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns].fillna(0)

        return self.best_model.predict(X)[0]

    def predict_proba(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Get probability distribution over all output materials."""
        if not self.trained:
            raise ValueError("Model not trained")

        X = pd.DataFrame([input_data])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns].fillna(0)

        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)[0]
        else:
            # Return one-hot for models without predict_proba
            pred = self.best_model.predict(X)[0]
            proba = np.zeros(len(self.best_model.classes_))
            proba[pred] = 1.0
            return proba

    def predict_top_k(
        self,
        input_data: Dict[str, Any],
        encoders: Dict,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top-k most likely output materials with probabilities.

        Returns list of dicts with:
        - output_material: Material code
        - probability: Predicted probability
        - rank: 1, 2, 3...
        """
        if not self.trained:
            raise ValueError("Model not trained")

        proba = self.predict_proba(input_data)
        classes = self.best_model.classes_

        # Get top-k indices
        top_indices = np.argsort(proba)[::-1][:k]

        # Decode material names
        output_encoder = encoders.get('Output_Material')

        results = []
        for rank, idx in enumerate(top_indices, 1):
            encoded_value = classes[idx]

            # Decode to material name
            if output_encoder and hasattr(output_encoder, 'inverse_transform'):
                try:
                    material_name = output_encoder.inverse_transform([encoded_value])[0]
                except:
                    material_name = f"Material_{encoded_value}"
            else:
                material_name = f"Material_{encoded_value}"

            results.append({
                'output_material': material_name,
                'output_material_encoded': int(encoded_value),
                'probability': float(proba[idx]),
                'probability_pct': round(float(proba[idx]) * 100, 2),
                'rank': rank
            })

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        for name in ['RandomForest_Classifier', 'XGBoost_Classifier', 'GradientBoosting_Classifier']:
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                return pd.DataFrame({
                    'Feature': self.feature_columns,
                    'Importance': self.models[name].feature_importances_
                }).sort_values('Importance', ascending=False)
        return pd.DataFrame()

    def save(self, path: str = None):
        """Save classifier artifacts."""
        if path is None:
            path = MODELS_DIR
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.best_model, os.path.join(path, 'output_classifier.joblib'))
        joblib.dump(self.models, os.path.join(path, 'all_classifiers.joblib'))
        joblib.dump(self.feature_columns, os.path.join(path, 'classifier_features.joblib'))
        joblib.dump(self.metrics, os.path.join(path, 'classifier_metrics.joblib'))

        print(f"Classifier saved to: {path}")

    def load(self, path: str = None):
        """Load classifier artifacts."""
        if path is None:
            path = MODELS_DIR

        self.best_model = joblib.load(os.path.join(path, 'output_classifier.joblib'))

        if os.path.exists(os.path.join(path, 'all_classifiers.joblib')):
            self.models = joblib.load(os.path.join(path, 'all_classifiers.joblib'))

        if os.path.exists(os.path.join(path, 'classifier_features.joblib')):
            self.feature_columns = joblib.load(os.path.join(path, 'classifier_features.joblib'))

        if os.path.exists(os.path.join(path, 'classifier_metrics.joblib')):
            self.metrics = joblib.load(os.path.join(path, 'classifier_metrics.joblib'))

        # Find best model name
        if self.metrics:
            self.best_model_name = max(
                self.metrics.keys(),
                key=lambda k: self.metrics[k].get('accuracy', 0)
            )

        self.trained = True
        print(f"Classifier loaded from: {path}")


def train_output_classifier(
    df: pd.DataFrame,
    encoders: Dict,
    save_path: str = None
) -> OutputMaterialClassifier:
    """
    Train the output material classification model.

    Given input material features, predicts which output material will be produced.
    """
    if save_path is None:
        save_path = MODELS_DIR

    # Check if Output_Material_Encoded exists
    if 'Output_Material_Encoded' not in df.columns:
        print("WARNING: Output_Material_Encoded not found. Cannot train classifier.")
        return None

    # Feature columns (same as yield model)
    feature_columns = [
        'Input_Material_Encoded',
        'Input_Specie_Encoded',
        'Input_Grade_Encoded',
        'Input_Plant_Encoded',
        'Input_Thickness',
        'Input_Length',
        'Input_Width',
        'Total_Input_BF'
    ]

    feature_columns = [c for c in feature_columns if c in df.columns]

    # Create and train classifier
    classifier = OutputMaterialClassifier()
    classifier.train(df, feature_columns, target_column='Output_Material_Encoded')

    # Save
    classifier.save(save_path)

    return classifier


if __name__ == "__main__":
    from .data_preparation import prepare_full_dataset

    csv_261 = os.path.join(DATA_DIR, "261.csv")
    csv_101 = os.path.join(DATA_DIR, "101.csv")
    if os.path.exists(csv_261) and os.path.exists(csv_101):
        print("Preparing data...")
        print("SAP Logic: 261 = INPUT (Goods Issue), 101 = OUTPUT (Goods Receipt)")
        df, encoders = prepare_full_dataset(csv_261, csv_101)

        # Train Yield Prediction Model (Regression)
        print("\n" + "=" * 70)
        print("TRAINING YIELD PREDICTION MODEL (Regression)")
        print("=" * 70)
        model = train_yield_model(df, encoders, MODELS_DIR)

        print("\nYield Model Comparison:")
        print(model.get_model_comparison())

        print("\nYield Model Feature Importance:")
        print(model.get_feature_importance())

        # Train Output Material Classifier (Classification)
        print("\n" + "=" * 70)
        print("TRAINING OUTPUT MATERIAL CLASSIFIER (Classification)")
        print("=" * 70)
        classifier = train_output_classifier(df, encoders, MODELS_DIR)

        if classifier:
            print("\nClassifier Feature Importance:")
            print(classifier.get_feature_importance())
    else:
        print(f"CSV files not found in {DATA_DIR} (need both 261.csv and 101.csv).")
