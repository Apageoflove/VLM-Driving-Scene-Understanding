"""Render helpers for the demo UI (CPU-only, no gradio import)."""
from __future__ import annotations

from typing import Any


def analysis_to_markdown(record: dict[str, Any]) -> str:
    """Format one analyzer record as Markdown for the demo's result panel."""
    if record.get("parse_status") == "error":
        lines = ["### 解析失败", ""]
        lines += [f"- {e}" for e in record.get("errors", [])]
        if record.get("raw"):
            lines += ["", "#### 模型原始输出", "", record["raw"]]
        return "\n".join(lines)

    vehicles = "、".join(f"{k}×{v}" for k, v in record.get("vehicles", {}).items()) or "无"
    signs = "、".join(f"{k}×{v}" for k, v in record.get("signs", {}).items()) or "无"
    warnings = ""
    if record.get("parse_status") == "partial":
        warnings = "\n\n> 注意：以下字段有缺失 —— " + "；".join(record.get("errors", []))
    return (
        f"### 车道线\n{record.get('lane') or '—'}\n\n"
        f"### 车辆与行人\n{vehicles}\n\n"
        f"### 交通标志/信号灯\n{signs}\n\n"
        f"### 潜在驾驶风险\n{record.get('risk') or '—'}{warnings}"
    )
