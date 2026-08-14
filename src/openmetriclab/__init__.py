from .classification import evaluate_classification
from .registration import evaluate_registration, landmark_tre
from .regression import evaluate_regression
from .segmentation import binary_segmentation_metrics, evaluate_segmentation

__all__ = [
    "binary_segmentation_metrics",
    "evaluate_classification",
    "evaluate_registration",
    "evaluate_regression",
    "evaluate_segmentation",
    "landmark_tre",
]
__version__ = "0.2.0"
