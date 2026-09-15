"""Structured schema for driving-scene analysis output.

The project's README promises structured understanding (four dimensions:
lane lines, vehicles, traffic signs, driving risk), but historically the
pipeline stored the model's free-form text verbatim — and even wrote
"[错误] ..." strings as if they were results. This module defines the
structured record every analysis must produce, plus validation with
explicit, per-field error messages so the analyzer's repair loop has
something to act on.

Deliberately stdlib-only (dataclass + manual checks): the parser,
schema, and evaluator all run on a machine with no torch installed —
CI and offline evaluation must not drag in the training stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

VEHICLE_CATEGORIES = ("小汽车", "卡车", "公交车", "摩托车", "自行车")
PEDESTRIAN_CATEGORY = "行人"
SIGN_CATEGORIES = ("交通锥",)

DIMENSIONS = ("lane", "vehicles", "signs", "risk")


@dataclass
class SceneAnalysis:
    """One image's structured analysis.

    Fields mirror the four numbered sections the LoRA training data uses
    (04_prepare_training_data.py), so a correctly formatted model reply
    maps 1:1 onto this record without lossy conversion.

    counts: category -> detected count, e.g. {"小汽车": 2, "行人": 1}.
    lane / signs / risk: free text as produced by the model; the schema
    guarantees presence and non-emptiness, not linguistic quality.
    """

    image: str
    lane: str = ""
    vehicles: dict[str, int] = field(default_factory=dict)
    signs: dict[str, int] = field(default_factory=dict)
    risk: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate(analysis: SceneAnalysis) -> list[str]:
    """Return a list of human-readable problems; empty list means valid.

    Checks are intentionally shallow: presence and types. Semantic checks
    (do the counts match ground truth?) belong to the evaluator, not the
    schema.
    """
    errors: list[str] = []
    if not analysis.image:
        errors.append("image: missing image name")
    if not analysis.lane or not analysis.lane.strip():
        errors.append("lane: empty lane description")
    if not isinstance(analysis.vehicles, dict):
        errors.append("vehicles: must be a category->count mapping")
    else:
        for category, count in analysis.vehicles.items():
            if not isinstance(count, int) or count < 0:
                errors.append(f"vehicles: count for {category!r} must be a non-negative int, got {count!r}")
    if not isinstance(analysis.signs, dict):
        errors.append("signs: must be a category->count mapping")
    if not analysis.risk or not analysis.risk.strip():
        errors.append("risk: empty risk description")
    return errors
