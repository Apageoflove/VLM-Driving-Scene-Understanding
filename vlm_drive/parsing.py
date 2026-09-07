"""Robust parsing of model outputs and ground-truth text into structured records.

Two model output formats are supported:

1. text format — the numbered four-section reply the LoRA adapter was
   trained on ("1. 车道线：...\n2. 车辆：...\n3. 交通标志/信号灯：...\n4. 潜在驾驶风险：...").
   Changing the prompt to demand JSON would fight the fine-tuning, so the
   trained format stays the default and is parsed with tolerant section
   matching instead of exact-string slicing.
2. json format — for the base model or future retraining: strict-JSON
   prompt, with fence stripping and bounded repair for the malformations
   vision models actually emit (trailing commas, prose preamble,
   truncated tail).

The ground-truth side reuses the same counting extraction: the training
data generator (04_prepare_training_data.py) emits counts as "2辆小汽车、
1辆卡车"-style text, and predictions trained on that format repeat the
pattern, so one extractor serves both sides of the evaluator.

Stdlib-only on purpose (see schema.py).
"""
from __future__ import annotations

import json
import re

from .schema import SceneAnalysis, validate

# "1. 车道线：" / "2、车辆:" — tolerant to full/half width punctuation and
# to the model occasionally dropping the dot.
_SECTION_RE = re.compile(
    r"[1234][.、]?\s*(车道线|车辆|交通标志(?:[/、]?信号灯)?|交通信号灯|信号灯|潜在驾驶风险)\s*[：:]",
)

# "2辆小汽车" / "3个行人" — the counting grammar shared by GT and predictions.
_COUNT_RE = re.compile(r"(\d+)\s*[辆个只条]\s*([一-龥]+)")
_PEDESTRIAN_HINT = "行人"
_VEHICLE_NONE_HINTS = ("无可见车辆", "没有车辆", "无车辆")
_VEHICLE_PRESENT_HINTS = ("前方可见", "辆")
_SIGN_NONE_HINTS = ("无交通标志", "没有交通标志", "无信号灯", "没有信号灯")

# Vehicle vocabulary the GT generator can emit (04_prepare_training_data.py
# name_cn map) — anything else inside a count match is still accepted, just
# not bucketed for scoring.
_KNOWN_VEHICLE_WORDS = ("小汽车", "卡车", "公交车", "摩托车", "自行车", "行人", "交通锥")


def _normalize_section_name(name: str) -> str:
    n = name.replace(" ", "")
    if n.startswith("车道"):
        return "lane"
    if n.startswith("车辆"):
        return "vehicles"
    if n.startswith("交通") or n.startswith("信号灯"):
        return "signs"
    if n.startswith("潜在") or "风险" in n:
        return "risk"
    return ""


def extract_counts(text: str) -> dict[str, int]:
    """Pull ``N<量词><类别>`` pairs out of free text.

    Repeated mentions of the same category are summed (the model sometimes
    re-lists vehicles across lines). Zero counts implied by "无行人" style
    phrases are handled by the caller via presence hints, not here.
    """
    counts: dict[str, int] = {}
    for match in _COUNT_RE.finditer(text or ""):
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        category = match.group(2).strip()
        # Trim trailing punctuation glued to the category by sloppy spacing.
        category = category.rstrip("。，,；;、")
        if not category:
            continue
        counts[category] = counts.get(category, 0) + number
    return counts


def parse_text_sections(text: str, image: str = "") -> SceneAnalysis:
    """Parse the trained numbered format into a SceneAnalysis.

    Missing sections surface as validation errors (see schema.validate)
    rather than exceptions: the analyzer's retry loop needs the error
    list, and batch runs must not die on one malformed reply.
    """
    analysis = SceneAnalysis(image=image, lane="", vehicles={}, signs={}, risk="", raw=text or "")
    if not text:
        return analysis

    matches = list(_SECTION_RE.finditer(text))
    for idx, match in enumerate(matches):
        section = _normalize_section_name(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if section == "lane":
            analysis.lane = body
        elif section == "vehicles":
            counts = extract_counts(body)
            if counts:
                analysis.vehicles = counts
            elif any(hint in body for hint in _VEHICLE_NONE_HINTS):
                analysis.vehicles = {}
            else:
                # Text without counts and without an explicit "none":
                # keep empty but remember the raw wording in lane-agnostic
                # field — vehicles stays {} and validation still passes.
                analysis.vehicles = {}
        elif section == "signs":
            counts = extract_counts(body)
            cone_like = {k: v for k, v in counts.items() if "锥" in k}
            if cone_like:
                analysis.signs = cone_like
            elif any(hint in body for hint in _SIGN_NONE_HINTS) or "无" in body:
                analysis.signs = {}
            else:
                analysis.signs = {"交通锥": 1} if "交通锥" in body else {}
        elif section == "risk":
            analysis.risk = body

    return analysis


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences and any leading prose before the JSON body."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text or "", re.S)
    if fenced:
        return fenced.group(1).strip()
    return (text or "").strip()


def _repair_common_json_errors(raw: str) -> str:
    """Best-effort repair of the malformations small VLMs emit.

    Deliberately conservative: each fix targets a pattern that cannot occur
    inside legal JSON strings' semantic layer without already being broken.
    """
    repaired = raw
    # Trailing comma before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Fullwidth quotes/colons used at key positions
    repaired = repaired.replace("：", ":").replace("，", ",")
    # Single-quoted keys/values -> double (only when clearly not apostrophes:
    # adjacent to braces/colons/commas)
    repaired = re.sub(r"(?<=[{,])\s*'([^']+?)'\s*:", lambda m: f' "{m.group(1)}":', repaired)
    repaired = re.sub(r":\s*'([^']*?)'", lambda m: f': "{m.group(1)}"', repaired)
    return repaired


def _truncate_to_balanced_json(raw: str) -> str | None:
    """Cut a truncated JSON object at the last complete field and close it."""
    if not raw.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    last_good = -1
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[: i + 1]
        if ch == "," and depth == 1:
            last_good = i
    if last_good >= 0:
        return raw[:last_good] + "}"
    return None


def parse_json_output(text: str, image: str = "") -> tuple[SceneAnalysis, list[str]]:
    """Parse a strict-JSON model reply into a SceneAnalysis.

    Returns (analysis, errors). errors is non-empty when the reply could
    not be salvaged into a schema-valid record — the analyzer retries
    with the errors appended to the prompt.
    """
    raw = strip_code_fences(text or "")
    if not raw.startswith("{"):
        start = raw.find("{")
        if start > 0:
            raw = raw[start:]
    if not raw:
        return SceneAnalysis(image=image, raw=text or ""), ["json: no JSON object found in output"]

    candidates = [raw, _repair_common_json_errors(raw), _truncate_to_balanced_json(raw) or raw,
                  _truncate_to_balanced_json(_repair_common_json_errors(raw)) or raw]
    data = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        return SceneAnalysis(image=image, raw=text or ""), ["json: output is not parseable JSON even after repair"]

    vehicles_raw = data.get("vehicles") or {}
    if isinstance(vehicles_raw, str):
        vehicles = extract_counts(vehicles_raw)
    elif isinstance(vehicles_raw, dict):
        vehicles = {str(k): int(v) for k, v in vehicles_raw.items() if isinstance(v, (int, float))}
    else:
        vehicles = {}

    signs_raw = data.get("signs") or {}
    if isinstance(signs_raw, str):
        signs = extract_counts(signs_raw)
    elif isinstance(signs_raw, dict):
        signs = {str(k): int(v) for k, v in signs_raw.items() if isinstance(v, (int, float))}
    else:
        signs = {}

    analysis = SceneAnalysis(
        image=image,
        lane=str(data.get("lane") or ""),
        vehicles=vehicles,
        signs=signs,
        risk=str(data.get("risk") or ""),
        raw=text or "",
    )
    errors = validate(analysis)
    return analysis, errors


def parse_ground_truth(text: str, image: str = "") -> SceneAnalysis:
    """Parse a GT sample (04 generator format) with the text-section parser.

    GT text is well-formed by construction, but running it through the same
    parser as predictions guarantees the evaluator compares like with like:
    any parsing quirk hits both sides equally instead of biasing scores.
    """
    return parse_text_sections(text, image=image)
