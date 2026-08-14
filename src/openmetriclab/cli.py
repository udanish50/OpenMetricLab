from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .regression import evaluate_regression
from .classification import evaluate_classification

def main():
    ap=argparse.ArgumentParser(prog='openmetriclab',description='Truth-vs-prediction evaluation for regression and classification CSV files.')
    ap.add_argument('csv'); ap.add_argument('--task',choices=['regression','classification'],required=True)
    ap.add_argument('--actual',required=True); ap.add_argument('--predicted',required=True)
    ap.add_argument('--prob-prefix',default='prob_'); args=ap.parse_args()
    with Path(args.csv).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    y=[r[args.actual] for r in rows]; p=[r[args.predicted] for r in rows]
    if args.task=='regression': result=evaluate_regression([float(x) for x in y],[float(x) for x in p])
    else:
        prob_cols=[c for c in rows[0] if c.startswith(args.prob_prefix)] if rows else []
        probs=[[float(r[c]) for c in prob_cols] for r in rows] if prob_cols else None
        classes=[c[len(args.prob_prefix):] for c in prob_cols] if prob_cols else None
        result=evaluate_classification(y,p,probs,classes)
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__': main()
