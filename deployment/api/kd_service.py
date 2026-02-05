"""KD Distribution Service - Provides KD material distribution with grade breakdown."""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class KDDistributionService:
    """Service for KD material distribution predictions."""

    def __init__(self, model_dir: str, model_service=None, preprocessor=None):
        """
        Initialize KD service with precomputed data and optional ML model.

        Args:
            model_dir: Path to directory containing historical_data_precomputed.joblib
            model_service: Optional ONNXModelService for ML yield prediction
            preprocessor: Optional Preprocessor for encoding inputs
        """
        self.model_dir = Path(model_dir)
        self.model_service = model_service
        self.preprocessor = preprocessor

        precomputed_path = self.model_dir / "historical_data_precomputed.joblib"

        if not precomputed_path.exists():
            raise FileNotFoundError(f"Precomputed data not found: {precomputed_path}")

        # Load precomputed data
        data = joblib.load(precomputed_path)
        self.material_stats = data.get('material_stats', {})
        self.grade_breakdown_data = data.get('grade_breakdown_data', {})
        self.year = data.get('year', 'unknown')

        print(f"KD Service loaded: {len(self.material_stats)} materials, year={self.year}")

    def _get_ml_yield_prediction(
        self,
        input_material: str,
        plant: str,
        input_bf: float,
        stats: Dict[str, Any]
    ) -> Optional[float]:
        """
        Use ONNX model to predict yield percentage.

        Args:
            input_material: KS input material code
            plant: Plant code
            input_bf: Input board feet
            stats: Pre-computed material statistics

        Returns:
            ML predicted yield percentage, or None if model unavailable
        """
        if self.model_service is None or self.preprocessor is None:
            return None

        try:
            # Build a mock request object with the fields preprocessor expects
            class MockRequest:
                pass

            request = MockRequest()
            request.input_material = input_material
            request.input_plant = plant
            request.input_thickness = stats.get('avg_thickness', 4.0)
            request.input_specie = stats.get('most_common_specie', 'SM')
            request.input_grade = stats.get('most_common_grade', '2C')
            request.input_length = stats.get('avg_length', 96.0)
            request.input_width = stats.get('avg_width', 8.0)
            request.total_input_bf = input_bf

            # Encode and predict
            features = self.preprocessor.encode(request)
            ml_yield = self.model_service.predict(features)

            # Clip to reasonable bounds (same as Streamlit)
            ml_yield = float(np.clip(ml_yield, 0, 150))

            return ml_yield

        except Exception as e:
            print(f"ML prediction failed: {e}")
            return None

    def get_kd_prediction(
        self,
        input_material: str,
        plant: str,
        input_bf: float,
        ml_yield_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get KD distribution prediction with grade breakdown.

        Uses ONNX ML model for yield prediction (matching Streamlit behavior),
        combined with historical KD distribution.

        Args:
            input_material: KS input material code
            plant: Plant code
            input_bf: Input board feet
            ml_yield_pct: Optional override for ML yield (if None, uses model)

        Returns:
            Dict with summary, KD outputs, and grade breakdowns
        """
        # Validate material exists
        if input_material not in self.material_stats:
            available = list(self.material_stats.keys())[:10]
            return {
                'error': f'Material not found: {input_material}. Available: {available}',
                'input_material': input_material,
                'plant': plant
            }

        # Validate plant exists for this material
        if plant not in self.material_stats[input_material]:
            available_plants = list(self.material_stats[input_material].keys())
            return {
                'error': f'Plant {plant} not found for {input_material}. Available: {available_plants}',
                'input_material': input_material,
                'plant': plant
            }

        stats = self.material_stats[input_material][plant]

        # Get historical yield for reference
        historical_yield_pct = stats['historical_yield_pct']

        # Use ML model for yield prediction (matching Streamlit behavior)
        if ml_yield_pct is not None:
            # Use provided override
            effective_yield = ml_yield_pct
            prediction_method = 'ML Model (override)'
        else:
            # Try ML model prediction
            ml_prediction = self._get_ml_yield_prediction(
                input_material, plant, input_bf, stats
            )
            if ml_prediction is not None:
                effective_yield = ml_prediction
                prediction_method = 'ML Model + Statistical Distribution'
            else:
                # Fallback to historical if ML unavailable
                effective_yield = historical_yield_pct
                prediction_method = 'Statistical (ML unavailable)'

        difference_pct = round(effective_yield - historical_yield_pct, 2)

        # Calculate predicted output using ML yield
        predicted_output_bf = input_bf * (effective_yield / 100)

        # Determine confidence level based on order count
        total_orders = stats['total_orders']
        if total_orders >= 50:
            ml_confidence = "HIGH"
        elif total_orders >= 20:
            ml_confidence = "MEDIUM"
        elif total_orders >= 10:
            ml_confidence = "LOW"
        else:
            ml_confidence = "VERY LOW"

        # Build summary
        summary = {
            'ml_predicted_yield_pct': round(effective_yield, 2),
            'historical_yield_pct': round(historical_yield_pct, 2),
            'difference_pct': difference_pct,
            'predicted_output_bf': round(predicted_output_bf, 2),
            'ml_confidence': ml_confidence,
            'historical_input_bf': round(stats['total_hist_input_bf'], 2),
            'historical_output_bf': round(stats['total_hist_output_bf'], 2),
            'historical_orders': total_orders,
            'prediction_method': prediction_method
        }

        # Build KD outputs with grade breakdown
        kd_outputs = []
        grade_data = self.grade_breakdown_data.get(input_material, {}).get(plant, {})

        for kd in stats['kd_distribution']:
            kd_material = kd['KD_Material']
            contrib_pct = kd['Contribution_Pct']
            expected_bf = predicted_output_bf * (contrib_pct / 100)

            # Get grade breakdown for this KD material
            grades = []
            kd_grades = grade_data.get(kd_material, [])

            for grade_info in kd_grades:
                # Calculate expected BF for this grade based on its percentage of KD
                grade_contrib_pct = grade_info.get('Pct_Of_KD', 0) * contrib_pct / 100
                grade_expected_bf = expected_bf * (grade_info.get('Pct_Of_KD', 0) / 100)

                grades.append({
                    'grade': grade_info.get('Grade', 'N/A'),
                    'contrib_pct': round(grade_contrib_pct, 2),
                    'expected_bf': round(grade_expected_bf, 2),
                    'orders': grade_info.get('Order_Count', 0),
                    'avg_length': grade_info.get('Avg_Length'),
                    'avg_width': grade_info.get('Avg_Width'),
                    'thickness': grade_info.get('Avg_Thickness')
                })

            kd_outputs.append({
                'kd_material': kd_material,
                'contrib_pct': round(contrib_pct, 2),
                'expected_bf': round(expected_bf, 2),
                'orders': kd['Order_Count'],
                'hist_bf': round(kd['Historical_BF_Output'], 2),
                'grades': grades
            })

        return {
            'input_material': input_material,
            'input_plant': plant,
            'input_bf': input_bf,
            'summary': summary,
            'kd_outputs': kd_outputs
        }

    def get_available_materials(self) -> List[str]:
        """Get list of available KS materials."""
        return list(self.material_stats.keys())

    def get_available_plants(self, material: str) -> List[str]:
        """Get list of available plants for a material."""
        if material not in self.material_stats:
            return []
        return list(self.material_stats[material].keys())
