from .regression import evaluate_regression
from .classification import evaluate_classification
from .segmentation import evaluate_segmentation, binary_segmentation_metrics
from .registration import evaluate_registration, landmark_tre
__all__=['evaluate_regression','evaluate_classification','evaluate_segmentation','binary_segmentation_metrics','evaluate_registration','landmark_tre']
__version__='0.1.0'
