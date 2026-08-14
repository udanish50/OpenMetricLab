from pathlib import Path
import hashlib,json

def test_manifest_hashes():
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/'examples/manifest.json').read_text())
    assert len(m['examples'])==8
    for ex in m['examples']:
        for item in ex['files']:
            p=root/item['path'];assert p.exists();assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
