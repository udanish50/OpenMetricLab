from pathlib import Path


def test_metric_source_does_not_import_external_metric_libraries():
    root = Path(__file__).resolve().parents[1] / "src" / "openmetriclab"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    blocked = ["sklearn", "scikit-learn", "skimage", "scikit-image", "scipy"]
    for token in blocked:
        assert token not in source.lower(), token
