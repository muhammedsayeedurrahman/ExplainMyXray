"""
Evaluation metrics for ExplainMyXray.

Computes exact match, strict/soft accuracy, precision/recall/F1, and
per-finding recall with synonym normalisation.

Usage as CLI:
    python evaluation/metrics.py --predictions results.json

Usage as library:
    from evaluation.metrics import evaluate_predictions
    metrics = evaluate_predictions(results)
"""

import argparse
import json
from collections import Counter
from typing import Dict, List

FINDING_SYNONYMS = {
    "cardiomegaly": ["enlarged heart", "cardiac enlargement", "enlarged cardiac silhouette"],
    "pleural effusion": ["fluid in pleural space", "pleural fluid"],
    "atelectasis": ["lung collapse", "partial collapse"],
    "pneumonia": ["lung infection", "pneumonic infiltrate", "consolidation"],
    "pneumothorax": ["collapsed lung", "air in pleural space"],
    "edema": ["pulmonary edema", "fluid overload"],
    "normal": ["no significant abnormalities detected", "no acute findings", "unremarkable"],
}

_SYN_MAP: Dict[str, str] = {}
for _canon, _syns in FINDING_SYNONYMS.items():
    _SYN_MAP[_canon] = _canon
    for _s in _syns:
        _SYN_MAP[_s] = _canon


def normalise_finding(finding: str) -> str:
    f = finding.lower().strip()
    if f in _SYN_MAP:
        return _SYN_MAP[f]
    for syn, canon in _SYN_MAP.items():
        if syn in f or f in syn:
            return canon
    return f


def extract_findings_from_report(text: str) -> List[str]:
    findings, in_findings = [], False
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("FINDINGS"):
            in_findings = True
            continue
        if line.upper().startswith(("LOCATIONS", "IMPRESSION")):
            in_findings = False
            continue
        if in_findings and line.startswith("- "):
            f = line.lstrip("- ").split("(")[0].strip().lower()
            if f and f not in ("nan", ""):
                findings.append(f)
        elif in_findings and "no significant" in line.lower():
            findings.append("normal")
    return findings if findings else ["normal"]


def evaluate_predictions(results: List[Dict]) -> Dict:
    """Evaluate a list of {gt_findings, prediction} dicts."""
    exact = strict = soft = total = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    pf_tp, pf_total = Counter(), Counter()

    for r in results:
        gt = set(normalise_finding(f) for f in r["gt_findings"] if f.strip())
        pred = set(normalise_finding(f) for f in extract_findings_from_report(r["prediction"]))
        total += 1

        if gt == pred:
            exact += 1
        if gt:
            overlap = len(gt & pred) / len(gt)
            if overlap >= 0.75:
                strict += 1
            if overlap >= 0.50:
                soft += 1
        else:
            if not pred or pred == {"normal"}:
                strict += 1
                soft += 1

        for f in gt & pred:
            tp[f] += 1
        for f in pred - gt:
            fp[f] += 1
        for f in gt - pred:
            fn[f] += 1
        for f in gt:
            pf_total[f] += 1
        for f in gt & pred:
            pf_tp[f] += 1

    ttp, tfp, tfn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    prec = ttp / (ttp + tfp) if ttp + tfp else 0.0
    rec = ttp / (ttp + tfn) if ttp + tfn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    per_finding = {}
    for f in sorted(pf_total, key=lambda x: pf_total[x], reverse=True):
        n = pf_total[f]
        t = pf_tp.get(f, 0)
        per_finding[f] = {"correct": t, "total": n, "recall": t / n if n else 0.0}

    return {
        "total_samples": total,
        "exact_match": exact,
        "exact_match_pct": round(exact / total * 100, 1) if total else 0.0,
        "strict_match": strict,
        "strict_match_pct": round(strict / total * 100, 1) if total else 0.0,
        "soft_match": soft,
        "soft_match_pct": round(soft / total * 100, 1) if total else 0.0,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "per_finding": per_finding,
    }


def print_report(m: Dict) -> None:
    print("=" * 60)
    print(f"  EVALUATION ({m['total_samples']} samples)")
    print("=" * 60)
    print(f"  Exact:    {m['exact_match']}/{m['total_samples']} ({m['exact_match_pct']}%)")
    print(f"  Strict:   {m['strict_match']}/{m['total_samples']} ({m['strict_match_pct']}%)")
    print(f"  Soft:     {m['soft_match']}/{m['total_samples']} ({m['soft_match_pct']}%)")
    print(f"  P / R / F1: {m['precision']} / {m['recall']} / {m['f1']}")
    print("-" * 60)
    for f, d in list(m["per_finding"].items())[:20]:
        print(f"  {f:30s} {d['correct']:3d}/{d['total']:3d}  ({d['recall']*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    with open(args.predictions) as f:
        results = json.load(f)
    metrics = evaluate_predictions(results)
    print_report(metrics)
    out = args.predictions.replace(".json", "_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
