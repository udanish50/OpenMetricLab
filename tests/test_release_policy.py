from pathlib import Path

import openmetriclab


def test_public_exports_are_sorted_and_complete():
    assert openmetriclab.__all__ == sorted(openmetriclab.__all__)
    assert set(openmetriclab.__all__) == {
        "binary_segmentation_metrics",
        "evaluate_classification",
        "evaluate_registration",
        "evaluate_regression",
        "evaluate_segmentation",
        "landmark_tre",
    }


def test_metric_runtime_has_no_external_metric_backends():
    root = Path(__file__).resolve().parents[1] / "src" / "openmetriclab"
    source = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.py"))
    lowered = source.lower()
    for token in ("sklearn", "scikit-learn", "skimage", "scikit-image", "scipy"):
        assert token not in lowered
