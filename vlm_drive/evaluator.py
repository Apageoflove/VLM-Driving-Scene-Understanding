"""Automatic four-dimension evaluator — replaces the manual scoring in the project plan.

The project plan (项目方案.md) literally says "人工打分" for the 50-image
evaluation set. This module turns that into a reproducible harness:

- parse_success_rate: share of predictions that produced a schema-valid
  record (the parser layer's headline metric);
- vehicle scoring: per-category count comparison against nuScenes ground
  truth, plus pedestrian hit/miss, reported as per-category exact-match
  rate, count MAE, and detection hit rate for "any vehicle present";
- sign scoring: traffic-cone presence hit/miss;
- text dimensions (lane, risk): coverage rate (non-empty after parsing) —
  semantic quality of free text is out of scope for automatic scoring and
  stays a human-review task, stated as such in the report.

Both prediction and ground-truth records go through the SAME parser
(parsing.parse_text_sections), so extraction quirks cannot bias one side.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .parsing import parse_ground_truth, parse_text_sections
from .schema import SceneAnalysis, validate

VEHICLE_BUCKET_WORDS = ("小汽车", "卡车", "公交车", "摩托车", "自行车")
PEDESTRIAN_WORD = "行人"
CONE_WORD = "交通锥"


def _as_analysis(record: Any, image: str) -> SceneAnalysis:
    """Accept either a structured dict (new pipeline) or raw text (legacy outputs)."""
    if isinstance(record, dict):
        return SceneAnalysis(
            image=str(record.get("image") or image),
            lane=str(record.get("lane") or ""),
            vehicles=dict(record.get("vehicles") or {}),
            signs=dict(record.get("signs") or {}),
            risk=str(record.get("risk") or ""),
            raw=str(record.get("raw") or ""),
        )
    # Legacy: old scripts stored the bare model string.
    return parse_text_sections(str(record), image=image)


def _bucket_vehicles(counts: dict[str, int]) -> dict[str, int]:
    bucketed: dict[str, int] = {}
    for category, count in counts.items():
        for word in VEHICLE_BUCKET_WORDS:
            if word in category:
                bucketed[word] = bucketed.get(word, 0) + count
                break
    return bucketed


def _has_pedestrian(counts: dict[str, int]) -> bool:
    return any(PEDESTRIAN_WORD in category for category, count in counts.items() if count > 0)


def _has_cone(signs: dict[str, int]) -> bool:
    return any(CONE_WORD in category for category, count in signs.items() if count > 0)


def evaluate(
    predictions: dict[str, Any],
    ground_truths: dict[str, Any],
) -> dict[str, Any]:
    """Score predictions against GT. Both are {image_name: record-or-text} maps."""
    images = sorted(set(ground_truths) & set(predictions))
    missing_pred = sorted(set(ground_truths) - set(predictions))

    per_image: list[dict[str, Any]] = []
    category_exact: dict[str, list[int]] = defaultdict(list)
    category_ae: dict[str, list[int]] = defaultdict(list)
    ped_hits: list[int] = []
    vehicle_presence_hits: list[int] = []
    cone_hits: list[int] = []
    lane_covered = 0
    risk_covered = 0
    parse_ok = 0

    for image in images:
        pred = _as_analysis(predictions[image], image)
        gt = _as_analysis(ground_truths[image], image) if isinstance(ground_truths[image], (str,)) else _as_analysis(ground_truths[image], image)

        if isinstance(ground_truths[image], str):
            gt = parse_ground_truth(ground_truths[image], image=image)

        pred_valid = not validate(pred)
        if pred_valid:
            parse_ok += 1

        pred_v = _bucket_vehicles(pred.vehicles)
        gt_v = _bucket_vehicles(gt.vehicles)

        row: dict[str, Any] = {"image": image, "parsed": pred_valid}

        vehicle_details: dict[str, dict[str, int]] = {}
        for word in VEHICLE_BUCKET_WORDS:
            p, g = pred_v.get(word, 0), gt_v.get(word, 0)
            category_exact[word].append(1 if p == g else 0)
            category_ae[word].append(abs(p - g))
            vehicle_details[word] = {"pred": p, "gt": g}
        row["vehicles"] = vehicle_details

        ped_p, ped_g = _has_pedestrian(pred.vehicles), _has_pedestrian(gt.vehicles)
        ped_hits.append(1 if ped_p == ped_g else 0)
        row["pedestrian"] = {"pred": ped_p, "gt": ped_g}

        presence_p, presence_g = bool(pred_v), bool(gt_v)
        vehicle_presence_hits.append(1 if presence_p == presence_g else 0)
        row["vehicle_presence"] = {"pred": presence_p, "gt": presence_g}

        cone_p, cone_g = _has_cone(pred.signs), _has_cone(gt.signs)
        cone_hits.append(1 if cone_p == cone_g else 0)
        row["cone_presence"] = {"pred": cone_p, "gt": cone_g}

        row["lane_covered"] = bool(pred.lane.strip())
        row["risk_covered"] = bool(pred.risk.strip())
        lane_covered += row["lane_covered"]
        risk_covered += row["risk_covered"]

        row["errors"] = validate(pred)
        per_image.append(row)

    n = len(images) or 1
    metrics: dict[str, Any] = {
        "images_evaluated": len(images),
        "images_missing_prediction": missing_pred,
        "parse_success_rate": round(parse_ok / n, 4),
        "lane_coverage": round(lane_covered / n, 4),
        "risk_coverage": round(risk_covered / n, 4),
        "vehicle_presence_accuracy": round(sum(vehicle_presence_hits) / n, 4),
        "pedestrian_hit_rate": round(sum(ped_hits) / n, 4),
        "cone_presence_hit_rate": round(sum(cone_hits) / n, 4),
        "vehicle_count": {
            word: {
                "exact_match_rate": round(sum(hits) / len(hits), 4) if hits else None,
                "count_mae": round(sum(errs) / len(errs), 4) if errs else None,
            }
            for word, hits in category_exact.items()
            for errs in [category_ae[word]]
        },
    }
    return {"metrics": metrics, "per_image": per_image}


def write_report(result: dict[str, Any], out_path: str | Path, *, title: str = "评估报告") -> Path:
    """Render metrics + per-image table as a Markdown report."""
    metrics, per_image = result["metrics"], result["per_image"]
    lines = [f"# {title}", ""]
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 评估图片数 | {metrics['images_evaluated']} |")
    if metrics["images_missing_prediction"]:
        lines.append(f"| 缺少预测的图片 | {len(metrics['images_missing_prediction'])} |")
    lines.append(f"| 结构化成功率 | {metrics['parse_success_rate']:.1%} |")
    lines.append(f"| 车道线字段覆盖率 | {metrics['lane_coverage']:.1%} |")
    lines.append(f"| 风险字段覆盖率 | {metrics['risk_coverage']:.1%} |")
    lines.append(f"| 有无车辆判断准确率 | {metrics['vehicle_presence_accuracy']:.1%} |")
    lines.append(f"| 行人有无命中率 | {metrics['pedestrian_hit_rate']:.1%} |")
    lines.append(f"| 交通锥有无命中率 | {metrics['cone_presence_hit_rate']:.1%} |")
    lines.append("")
    lines.append("## 车辆计数（按类别，对比 nuScenes 标注）")
    lines.append("")
    lines.append("| 类别 | 计数完全一致率 | 计数平均绝对误差 |")
    lines.append("|---|---|---|")
    for word, stat in metrics["vehicle_count"].items():
        exact = "—" if stat["exact_match_rate"] is None else f"{stat['exact_match_rate']:.1%}"
        mae = "—" if stat["count_mae"] is None else f"{stat['count_mae']:.2f}"
        lines.append(f"| {word} | {exact} | {mae} |")
    lines.append("")
    lines.append("## 逐图明细")
    lines.append("")
    lines.append("| 图片 | 结构化 | 车辆(预测/标注) | 行人 | 交通锥 |")
    lines.append("|---|---|---|---|---|")
    for row in per_image:
        vehicles = "、".join(
            f"{word}:{detail['pred']}/{detail['gt']}"
            for word, detail in row["vehicles"].items()
            if detail["pred"] or detail["gt"]
        ) or "—"
        ped = row["pedestrian"]
        cone = row["cone_presence"]
        lines.append(
            f"| {row['image']} | {'是' if row['parsed'] else '否'} | {vehicles} "
            f"| {'有' if ped['pred'] else '无'}/{'有' if ped['gt'] else '无'} "
            f"| {'有' if cone['pred'] else '无'}/{'有' if cone['gt'] else '无'} |"
        )
    lines.append("")
    lines.append("说明：车道线与风险为自由文本维度，本报告只统计覆盖率；语义质量仍需人工复核。")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def evaluate_files(predictions_path: str | Path, ground_truth_path: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    """Load two JSON files, evaluate, optionally write report + metrics."""
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    ground_truths = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    result = evaluate(predictions, ground_truths)
    if report_path:
        write_report(result, report_path)
        metrics_path = Path(report_path).with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    return result
