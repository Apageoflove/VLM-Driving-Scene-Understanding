"""Single source of the project's domain vocabulary.

Everything that needs category words imports from here, so the vocabulary
cannot drift between the GT builder, the parser, and the evaluator:

- NUSCENES_CATEGORY_CN is the nuScenes -> Chinese display-name map the GT
  text is generated from (mirrors 04_prepare_training_data.py);
- VEHICLE_CN / PEDESTRIAN_CN / SIGN_CN are derived from it, not re-listed;
- SURFACE_ALIASES maps the wording the fine-tuned model actually emits
  (汽车/车辆/巴士/单车...) onto those canonical buckets. Observed in the
  real lora_eval_results.json replies; extend here when new drift shows up.
"""
from __future__ import annotations

NUSCENES_CATEGORY_CN: dict[str, str] = {
    "vehicle.car": "小汽车",
    "vehicle.truck": "卡车",
    "vehicle.bus.rigid": "公交车",
    "vehicle.motorcycle": "摩托车",
    "vehicle.bicycle": "自行车",
    "human.pedestrian.adult": "行人",
    "movable_object.trafficcone": "交通锥",
}

# Derived buckets — do not edit by hand; add categories above instead.
VEHICLE_CN: tuple[str, ...] = tuple(
    name for key, name in NUSCENES_CATEGORY_CN.items() if key.startswith("vehicle.")
)
PEDESTRIAN_CN: tuple[str, ...] = tuple(
    name for key, name in NUSCENES_CATEGORY_CN.items() if key.startswith("human.")
)
SIGN_CN: tuple[str, ...] = tuple(
    name for key, name in NUSCENES_CATEGORY_CN.items() if key.startswith("movable_object.")
)

# Model-output wording -> canonical bucket (targets must exist in the buckets
# above; enforced by tests/test_vlm_drive.py).
SURFACE_ALIASES: dict[str, str] = {
    "汽车": "小汽车",
    "车辆": "小汽车",
    "巴士": "公交车",
    "单车": "自行车",
    "锥桶": "交通锥",
    "锥": "交通锥",
    "路人": "行人",
}


def canonicalize(category: str) -> str:
    """Return the canonical bucket name for a surface category, or ''."""
    for canonical in (*VEHICLE_CN, *PEDESTRIAN_CN, *SIGN_CN):
        if canonical in category:
            return canonical
    for alias, canonical in SURFACE_ALIASES.items():
        if alias in category:
            return canonical
    return ""


def is_object_category(category: str) -> bool:
    """True when a counted 'category' names a real object, not prose.

    Used by the parser to drop measurement noise (一辆车距离约为50米 -> the
    numeral grammar grabs '车距离约为' as a category).
    """
    return canonicalize(category) != ""
