"""Post-hoc calibration for confidence scores using temperature scaling."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CalibrationData:
    """Data for calibration."""
    confidences: List[float]
    correctness: List[bool]  # 1 if correct, 0 if incorrect
    task_family: Optional[str] = None


class TemperatureScaler:
    """Post-hoc temperature scaling for calibration.
    
    Based on Guo et al. "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599
    """
    
    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False
    
    def fit(self, confidences: np.ndarray, correctness: np.ndarray) -> float:
        """Fit temperature parameter to minimize NLL.
        
        Args:
            confidences: Array of confidence scores [0, 1]
            correctness: Array of binary correctness labels
        
        Returns:
            Optimal temperature
        """
        def nll(temp):
            """Negative log likelihood."""
            scaled = self._apply_temperature(confidences, temp)
            # Clip to avoid log(0)
            scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
            
            # Binary cross-entropy
            loss = -np.mean(
                correctness * np.log(scaled) +
                (1 - correctness) * np.log(1 - scaled)
            )
            return loss
        
        # Optimize temperature
        result = minimize(
            nll,
            x0=1.0,
            method='BFGS',
            options={'maxiter': 100}
        )
        
        self.temperature = float(result.x[0])
        self.is_fitted = True
        
        return self.temperature
    
    def _apply_temperature(
        self,
        confidences: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to confidences."""
        # For binary classification, we scale logits
        # Convert confidence to logit
        logits = np.log(confidences / (1 - confidences + 1e-10))
        
        # Scale by temperature
        scaled_logits = logits / temperature
        
        # Convert back to probability
        scaled_conf = 1 / (1 + np.exp(-scaled_logits))
        
        return scaled_conf
    
    def calibrate(self, confidence: float) -> float:
        """Apply calibration to a single confidence score."""
        if not self.is_fitted:
            return confidence
        
        return float(self._apply_temperature(
            np.array([confidence]),
            self.temperature
        )[0])
    
    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Apply calibration to a batch of confidence scores."""
        if not self.is_fitted:
            return confidences
        
        return self._apply_temperature(
            np.array(confidences),
            self.temperature
        ).tolist()


class MultiDomainCalibrator:
    """Multi-domain temperature scaling for heterogeneous tasks.
    
    Maintains separate calibrators for different task families.
    """
    
    def __init__(self, task_families: List[str]):
        """Initialize with task families.
        
        Args:
            task_families: List of task family names (e.g., 'coding', 'research')
        """
        self.task_families = task_families
        self.calibrators: Dict[str, TemperatureScaler] = {
            family: TemperatureScaler()
            for family in task_families
        }
        self.global_calibrator = TemperatureScaler()
    
    def fit(self, calibration_data: List[CalibrationData]):
        """Fit calibrators for each task family.
        
        Args:
            calibration_data: List of calibration data per task family
        """
        # Fit per-family calibrators
        for data in calibration_data:
            if data.task_family and data.task_family in self.calibrators:
                confidences = np.array(data.confidences)
                correctness = np.array(data.correctness, dtype=float)
                
                if len(confidences) > 0:
                    self.calibrators[data.task_family].fit(confidences, correctness)
        
        # Fit global calibrator on all data
        all_confidences = []
        all_correctness = []
        for data in calibration_data:
            all_confidences.extend(data.confidences)
            all_correctness.extend(data.correctness)
        
        if all_confidences:
            self.global_calibrator.fit(
                np.array(all_confidences),
                np.array(all_correctness, dtype=float)
            )
    
    def calibrate(
        self,
        confidence: float,
        task_family: Optional[str] = None
    ) -> float:
        """Calibrate a confidence score.
        
        Args:
            confidence: Raw confidence score
            task_family: Task family (uses global if not specified)
        
        Returns:
            Calibrated confidence score
        """
        if task_family and task_family in self.calibrators:
            calibrator = self.calibrators[task_family]
            if calibrator.is_fitted:
                return calibrator.calibrate(confidence)
        
        # Fall back to global calibrator
        return self.global_calibrator.calibrate(confidence)
    
    def save(self, filepath: str):
        """Save calibrators to disk."""
        data = {
            "task_families": self.task_families,
            "calibrators": {
                family: {
                    "temperature": cal.temperature,
                    "is_fitted": cal.is_fitted
                }
                for family, cal in self.calibrators.items()
            },
            "global": {
                "temperature": self.global_calibrator.temperature,
                "is_fitted": self.global_calibrator.is_fitted
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load calibrators from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.task_families = data["task_families"]
        
        for family, cal_data in data["calibrators"].items():
            self.calibrators[family] = TemperatureScaler()
            self.calibrators[family].temperature = cal_data["temperature"]
            self.calibrators[family].is_fitted = cal_data["is_fitted"]
        
        self.global_calibrator = TemperatureScaler()
        self.global_calibrator.temperature = data["global"]["temperature"]
        self.global_calibrator.is_fitted = data["global"]["is_fitted"]


class ThreeLevelCalibrator:
    """Three-level calibration for the agentic system.
    
    Calibrates:
    1. Base model confidence
    2. Router activation confidence
    3. Final answer confidence
    """
    
    def __init__(self, task_families: List[str]):
        self.base_model_calibrator = MultiDomainCalibrator(task_families)
        self.router_calibrator = MultiDomainCalibrator(task_families)
        self.final_answer_calibrator = MultiDomainCalibrator(task_families)
    
    def fit_base_model(self, calibration_data: List[CalibrationData]):
        """Fit base model calibrator."""
        self.base_model_calibrator.fit(calibration_data)
    
    def fit_router(self, calibration_data: List[CalibrationData]):
        """Fit router calibrator."""
        self.router_calibrator.fit(calibration_data)
    
    def fit_final_answer(self, calibration_data: List[CalibrationData]):
        """Fit final answer calibrator."""
        self.final_answer_calibrator.fit(calibration_data)
    
    def calibrate_base_model(
        self,
        confidence: float,
        task_family: Optional[str] = None
    ) -> float:
        """Calibrate base model confidence."""
        return self.base_model_calibrator.calibrate(confidence, task_family)
    
    def calibrate_router(
        self,
        confidence: float,
        task_family: Optional[str] = None
    ) -> float:
        """Calibrate router confidence."""
        return self.router_calibrator.calibrate(confidence, task_family)
    
    def calibrate_final_answer(
        self,
        confidence: float,
        task_family: Optional[str] = None
    ) -> float:
        """Calibrate final answer confidence."""
        return self.final_answer_calibrator.calibrate(confidence, task_family)
    
    def save(self, directory: str):
        """Save all calibrators."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        self.base_model_calibrator.save(str(dir_path / "base_model.json"))
        self.router_calibrator.save(str(dir_path / "router.json"))
        self.final_answer_calibrator.save(str(dir_path / "final_answer.json"))
    
    def load(self, directory: str):
        """Load all calibrators."""
        dir_path = Path(directory)
        
        self.base_model_calibrator.load(str(dir_path / "base_model.json"))
        self.router_calibrator.load(str(dir_path / "router.json"))
        self.final_answer_calibrator.load(str(dir_path / "final_answer.json"))


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error.
    
    Args:
        confidences: Array of confidence scores
        correctness: Array of binary correctness labels
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if i == n_bins - 1:  # Include upper bound in last bin
            in_bin = in_bin | (confidences == 1.0)
        
        if not np.any(in_bin):
            continue
        
        # Calculate accuracy and confidence in bin
        bin_accuracy = np.mean(correctness[in_bin])
        bin_confidence = np.mean(confidences[in_bin])
        bin_count = np.sum(in_bin)
        
        # Add weighted difference to ECE
        ece += (bin_count / total_samples) * abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def compute_brier_score(
    confidences: np.ndarray,
    correctness: np.ndarray
) -> float:
    """Compute Brier score.
    
    Args:
        confidences: Array of confidence scores
        correctness: Array of binary correctness labels
    
    Returns:
        Brier score (lower is better)
    """
    return float(np.mean((confidences - correctness) ** 2))
