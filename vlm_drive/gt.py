"""Build evaluation ground truth for the 50 eval images from local nuScenes tables.

Reuses 04_prepare_training_data.py's conversion logic (same category map,
same output text format) so GT and training data stay byte-compatible with
the parser. Eval images include non-keyframe sweeps; a sweep's sample_token
points at its owning keyframe sample, whose annotations are the nearest
ground truth (<= 0.5s apart in nuScenes) — noted honestly in the report.

Usage:
    python -m vlm_drive gt --nuscenes data/v1.0-trainval --images data/eval_images --out data/eval_gt.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .categories import NUSCENES_CATEGORY_CN

NAME_CN = NUSCENES_CATEGORY_CN
VEHICLE_NAMES = {k for k in NUSCENES_CATEGORY_CN if k.startswith("vehicle.")}
PED_NAMES = {k for k in NUSCENES_CATEGORY_CN if k.startswith("human.")}
SIGN_NAMES = {k for k in NUSCENES_CATEGORY_CN if k.startswith("movable_object.")}


def build_gt(nuscenes_dir: str | Path, images_dir: str | Path, *, min_visibility: int = 1) -> dict[str, str]:
    """min_visibility: nuScenes visibility tier 1-4 (1=0-40% ... 4=80-100%).
    Annotations are 360°-ring truth while the model sees only CAM_FRONT;
    raising the tier drops heavily-occluded objects and narrows that gap.
    """

    base = Path(nuscenes_dir)
    tables = {}
    vis_token_to_tier = {}
    vis_path = base / "visibility.json"
    if vis_path.exists():
        entries = json.loads(vis_path.read_text(encoding="utf-8"))
        def _lower_bound(entry: dict) -> int:
            m = re.search(r"(\d+)", entry.get("description", ""))
            return int(m.group(1)) if m else 0
        for tier, entry in enumerate(sorted(entries, key=_lower_bound), start=1):
            vis_token_to_tier[entry["token"]] = tier

    for name in ("sample_data", "sample_annotation", "instance", "category"):
        tables[name] = json.loads((base / f"{name}.json").read_text(encoding="utf-8"))

    filename_to_sample = {Path(s["filename"]).name: s["sample_token"] for s in tables["sample_data"]}
    inst_to_cat = {i["token"]: i["category_token"] for i in tables["instance"]}
    cat_to_name = {c["token"]: c["name"] for c in tables["category"]}
    ann_index: dict[str, list] = {}
    for ann in tables["sample_annotation"]:
        ann_index.setdefault(ann["sample_token"], []).append(ann)

    gt: dict[str, str] = {}
    for image_path in sorted(Path(images_dir).glob("*.jpg")):
        token = filename_to_sample.get(image_path.name)
        if not token:
            continue
        count_dict: dict[str, int] = {}
        for ann in ann_index.get(token, []):
            if vis_token_to_tier and vis_token_to_tier.get(ann.get("visibility_token", ""), 4) < min_visibility:
                continue
            name = cat_to_name.get(inst_to_cat.get(ann["instance_token"], ""), "")
            if name:
                count_dict[name] = count_dict.get(name, 0) + 1

        vehicles = [f"{n}辆{NAME_CN[e]}" for e, n in count_dict.items() if e in VEHICLE_NAMES]
        pedestrians = [f"{n}个{NAME_CN[e]}" for e, n in count_dict.items() if e in PED_NAMES]
        cones = [NAME_CN[e] for e, n in count_dict.items() if e in SIGN_NAMES]

        gt[image_path.name] = (
            "1. 车道线：根据图片判断车道线数量和类型。\n"
            f"2. 车辆：{'、'.join(vehicles) if vehicles else '前方无可见车辆'}。{'、'.join(pedestrians) if pedestrians else '无行人'}。\n"
            f"3. 交通标志/信号灯：{'、'.join(cones) if cones else '无交通标志或信号灯'}。\n"
            "4. 潜在驾驶风险：根据场景判断潜在风险。"
        )
    return gt
