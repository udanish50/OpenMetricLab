from __future__ import annotations
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'examples/manifest.json').read_text(encoding='utf-8'))
assert len(m['examples'])==8
count=0
for ex in m['examples']:
    for item in ex['files']:
        p=ROOT/item['path'];assert p.is_file(),p
        got=hashlib.sha256(p.read_bytes()).hexdigest();assert got==item['sha256'],(p,got,item['sha256']);count+=1
# semantic checks
reg=ROOT/'examples/data/regression_diabetes.csv'
with reg.open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
assert len(rows)>100 and {'actual','ridge','random_forest'}<=set(rows[0])
print(f'PASS OpenMetricLab examples: {len(m["examples"])} examples · {count} hashed files')
