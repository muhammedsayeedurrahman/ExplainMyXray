"""
Evaluation metrics for ExplainMyXray.

Computes:
  - Text: exact match, strict/soft accuracy, precision/recall/F1, per-finding recall
  - Spatial: IoU (Intersection over Union), mean IoU, per-finding spatial accuracy

Usage as CLI:
    python evaluation/metrics.py --predictions results.json

Usage as library:
    from evaluation.metrics import evaluate_predictions
    metrics = evaluate_predictions(results)

results.json format:
    [
      {
        "gt_findings": ["cardiomegaly", "pleural effusion"],
        "prediction": "FINDINGS:\n- Cardiomegaly ...\nLOCATIONS:\n...",
        "gt_bboxes": [{"xmin": 150, "ymin": 120, "xmax": 380, "ymax": 400, "label": "cardiomegaly"}],
        "pred_bboxes": [{"xmin": 155, "ymin": 125, "xmax": 375, "ymax": 395, "label": "Region 1"}]
      }
    ]

To generate results.json, run inference on a test set and save predictions:
    python evaluation/generate_predictions.py --images /path/to/test --output results.json
"""

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Spatial metrics (IoU)
# ---------------------------------------------------------------------------
def compute_iou(box_a: Dict, box_b: Dict) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Each box is a dict with keys: xmin, ymin, xmax, ymax.
    """
    x1 = max(box_a["xmin"], box_b["xmin"])
    y1 = max(box_a["ymin"], box_b["ymin"])
    x2 = min(box_a["xmax"], box_b["xmax"])
    y2 = min(box_a["ymax"], box_b["ymax"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    area_b = (box_b["xmax"] - box_b["xmin"]) * (box_b["ymax"] - box_b["ymin"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes_greedy(
    gt_boxes: List[Dict], pred_boxes: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """Greedy bipartite matching of predicted boxes to ground-truth boxes.

    Returns a list of match records with gt_idx, pred_idx, and iou.
    """
    if not gt_boxes or not pred_boxes:
        return []

    # Compute full IoU matrix
    iou_matrix = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou_matrix.append((compute_iou(gt, pred), gi, pi))
    iou_matrix.sort(reverse=True)

    matched_gt, matched_pred, matches = set(), set(), []
    for iou_val, gi, pi in iou_matrix:
        if iou_val < iou_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matches.append({"gt_idx": gi, "pred_idx": pi, "iou": iou_val})
        matched_gt.add(gi)
        matched_pred.add(pi)

    return matches


def evaluate_spatial(results: List[Dict], iou_threshold: float = 0.5) -> Optional[Dict]:
    """Evaluate bounding-box predictions using IoU-based metrics.

    Expects each result dict to optionally contain:
        - gt_bboxes: list of {xmin, ymin, xmax, ymax, label}
        - pred_bboxes: list of {xmin, ymin, xmax, ymax, label}

    Returns None if no spatial data is present.
    """
    all_ious = []
    total_gt = total_pred = total_tp = 0
    samples_with_spatial = 0

    for r in results:
        gt_boxes = r.get("gt_bboxes", [])
        pred_boxes = r.get("pred_bboxes", [])
        if not gt_boxes and not pred_boxes:
            continue

        samples_with_spatial += 1
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matches = match_boxes_greedy(gt_boxes, pred_boxes, iou_threshold)
        total_tp += len(matches)
        all_ious.extend(m["iou"] for m in matches)

    if samples_with_spatial == 0:
        return None

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "samples_with_spatial": samples_with_spatial,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "true_positives": total_tp,
        "mean_iou": round(mean_iou, 3),
        "spatial_precision": round(precision, 3),
        "spatial_recall": round(recall, 3),
        "spatial_f1": round(f1, 3),
        "iou_threshold": iou_threshold,
    }


# ---------------------------------------------------------------------------
# Text evaluation
# ---------------------------------------------------------------------------
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

    metrics = {
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

    # Add spatial metrics if bbox data is present
    spatial = evaluate_spatial(results)
    if spatial:
        metrics["spatial"] = spatial

    return metrics


def print_report(m: Dict) -> None:
    print("=" * 60)
    print(f"  TEXT EVALUATION ({m['total_samples']} samples)")
    print("=" * 60)
    print(f"  Exact:    {m['exact_match']}/{m['total_samples']} ({m['exact_match_pct']}%)")
    print(f"  Strict:   {m['strict_match']}/{m['total_samples']} ({m['strict_match_pct']}%)")
    print(f"  Soft:     {m['soft_match']}/{m['total_samples']} ({m['soft_match_pct']}%)")
    print(f"  P / R / F1: {m['precision']} / {m['recall']} / {m['f1']}")
    print("-" * 60)
    for f, d in list(m["per_finding"].items())[:20]:
        print(f"  {f:30s} {d['correct']:3d}/{d['total']:3d}  ({d['recall']*100:5.1f}%)")

    if "spatial" in m:
        s = m["spatial"]
        print()
        print("=" * 60)
        print(f"  SPATIAL EVALUATION ({s['samples_with_spatial']} samples, IoU >= {s['iou_threshold']})")
        print("=" * 60)
        print(f"  Mean IoU:   {s['mean_iou']}")
        print(f"  Precision:  {s['spatial_precision']}")
        print(f"  Recall:     {s['spatial_recall']}")
        print(f"  F1:         {s['spatial_f1']}")
        print(f"  GT boxes:   {s['total_gt_boxes']}")
        print(f"  Pred boxes: {s['total_pred_boxes']}")
        print(f"  Matched:    {s['true_positives']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ExplainMyXray predictions (text + spatial)"
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to results.json (see module docstring for format)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for spatial matching (default: 0.5)")
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
