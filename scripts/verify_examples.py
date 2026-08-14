from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "examples/manifest.json").read_text(encoding="utf-8"))
assert len(manifest["examples"]) == 8
count = 0
for example in manifest["examples"]:
    for item in example["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"], (path, digest, item["sha256"])
        count += 1

regression = ROOT / "examples/data/regression_diabetes.csv"
with regression.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) > 100
assert {"actual", "ridge", "random_forest"} <= set(rows[0])
print(f'PASS OpenMetricLab examples: {len(manifest["examples"])} examples · {count} hashed files')
