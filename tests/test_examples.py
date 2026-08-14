import hashlib
import json
from pathlib import Path


def test_manifest_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "examples/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["examples"]) == 8
    for example in manifest["examples"]:
        for item in example["files"]:
            path = root / item["path"]
            assert path.exists()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
