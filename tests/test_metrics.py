import math

import numpy as np

from openmetriclab import (
    evaluate_classification,
    evaluate_registration,
    evaluate_regression,
    evaluate_segmentation,
    landmark_tre,
)
from openmetriclab.registration import normalized_mutual_information, structural_similarity


def close(a, b, tol=1e-10):
    assert a is not None
    assert abs(a - b) <= tol


def test_regression_perfect_and_hand_calculated_case():
    perfect = evaluate_regression([1, 2, 3], [1, 2, 3])
    assert perfect["rmse"] == 0
    assert perfect["r2"] == 1

    result = evaluate_regression([1, 2, 3], [2, 2, 4])
    close(result["mae"], 2 / 3)
    close(result["mse"], 2 / 3)
    close(result["rmse"], math.sqrt(2 / 3))
    close(result["bias"], 2 / 3)
    close(result["r2"], 0.0)


def test_regression_zero_target_percentage_policy():
    result = evaluate_regression([0, 10, 20], [2, 9, 22])
    close(result["mape_coverage"], 2 / 3)
    close(result["mape_percent"], 10.0)


def test_classification_confusion_and_hand_metrics():
    y = [0, 0, 1, 1]
    p = [0, 1, 1, 1]
    result = evaluate_classification(y, p, classes=[0, 1])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    close(result["accuracy"], 0.75)
    close(result["balanced_accuracy"], 0.75)
    close(result["cohen_kappa"], 0.5)
    close(result["mcc"], 1 / math.sqrt(3))


def test_classification_probability_metrics_and_ties():
    y = [0, 1, 0, 1]
    p = [0, 1, 0, 1]
    probs = [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.5, 0.5]]
    result = evaluate_classification(y, p, probs, classes=[0, 1])
    close(result["roc_auc"], 0.875)
    assert 0 <= result["average_precision"] <= 1
    assert result["probabilities_normalized"] is False


def test_probability_rows_are_normalized_explicitly():
    result = evaluate_classification(
        [0, 1],
        [0, 1],
        probabilities=[[8, 2], [1, 9]],
        classes=[0, 1],
    )
    assert result["probabilities_normalized"] is True
    assert result["roc_auc"] == 1.0


def test_multiclass_mcc_perfect():
    result = evaluate_classification([0, 1, 2, 0], [0, 1, 2, 0])
    assert result["accuracy"] == 1.0
    assert result["mcc"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_segmentation_perfect_and_shifted():
    a = np.zeros((16, 16), dtype=int)
    a[4:12, 4:12] = 1
    perfect = evaluate_segmentation(a, a)
    assert perfect["macro"]["dice"] == 1
    assert perfect["macro"]["iou"] == 1
    assert perfect["macro"]["hausdorff"] == 0

    b = np.roll(a, shift=1, axis=1)
    shifted = evaluate_segmentation(a, b)
    assert 0 < shifted["macro"]["dice"] < 1
    assert shifted["macro"]["hausdorff"] == 1


def test_segmentation_both_empty_policy():
    a = np.zeros((8, 8), dtype=int)
    result = evaluate_segmentation(a, a, labels=[1])
    row = result["per_class"][0]
    assert row["dice"] == 1.0
    assert row["iou"] == 1.0
    assert row["surface_status"] == "both_empty"


def test_registration_perfect():
    a = np.arange(256, dtype=float).reshape(16, 16)
    result = evaluate_registration(a, a)
    assert result["rmse"] == 0
    assert result["ssim"] == 1
    assert math.isinf(result["psnr_db"])
    close(result["nmi"], 2.0)


def test_registration_similarity_helpers():
    a = np.arange(256, dtype=float).reshape(16, 16)
    b = a.copy()
    close(structural_similarity(a, b, data_range=255), 1.0)
    close(normalized_mutual_information(a, b, bins=32), 2.0)


def test_tre():
    result = landmark_tre([[0, 0], [1, 1]], [[3, 4], [1, 1]])
    close(result["mean_tre"], 2.5)
    close(result["max_tre"], 5.0)
