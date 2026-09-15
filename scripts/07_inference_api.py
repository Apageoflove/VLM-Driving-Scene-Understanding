"""
07_inference_api.py — 推理 API 端（v0.2：迁移到 vlm_drive 包之上）

v1 的缺陷（本版本修复）：
- 自带一份 80 行模型循环（与 01/03/06 三份重复的第四份）
- 输出裸文本，出错时把 "[错误] ..." 当结果写进 results
- 批量跑挂了只能从零重来
- 没有任何评测能力

现在 DrivingSceneAnalyzer 直接来自 vlm_drive（结构化记录、断点续跑、
repetition_penalty=1.2 保留 v1 的防循环修复）。本模块只保留 API 端特有的
东西：单张/批量入口 + 批量后自动评测。

用法：
    env/bin/python scripts/07_inference_api.py            # 批量推理 + 自动评测
    或在其他代码中 from vlm_drive.analyzer import DrivingSceneAnalyzer
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# vlm_drive.DrivingSceneAnalyzer 的构造签名 (model_path, lora_path=None, *, ...)
# 与 v1 完全兼容：DrivingSceneAnalyzer(model_path, lora_path) 照常工作；
# analyze() 的返回从裸文本变为结构化记录（parse_status/lane/vehicles/...），
# 需要裸文本用 generate_text(image)。
from vlm_drive.analyzer import DrivingSceneAnalyzer  # noqa: F401  (re-export)

from vlm_drive.evaluator import evaluate_files

DEFAULT_MODEL = _project_root / "models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_LORA = _project_root / "models" / "lora_checkpoint"
DEFAULT_EVAL_DIR = _project_root / "data" / "eval_images"


def run_eval_batch(
    output_file,
    gt_file=None,
    model_path=DEFAULT_MODEL,
    lora_path=DEFAULT_LORA,
    images_dir=DEFAULT_EVAL_DIR,
):
    """批量推理（断点续跑）→ 有真值就直接自动评测出报告。

    第一版跑完 50 张只剩一个裸文本 JSON，评分靠人工；现在一条链路出
    results + metrics + Markdown 报告。gt_file 缺省取 data/eval_gt.json，
    不存在则跳过评测（只出结构化结果）。
    """
    analyzer = DrivingSceneAnalyzer(
        model_path,
        lora_path,
        repetition_penalty=1.2,  # v1 的防循环修复，行为保持
    )
    analyzer.analyze_folder(images_dir, output_file)

    gt = Path(gt_file) if gt_file else _project_root / "data" / "eval_gt.json"
    if not gt.exists():
        print(f"未找到真值文件 {gt}，跳过自动评测（可用 vlm_drive gt 命令生成）")
        return None

    report = Path(output_file).with_suffix(".report.md")
    result = evaluate_files(output_file, gt, report_path=report)
    print(f"自动评测完成，报告：{report}")
    return result


if __name__ == "__main__":
    out = _project_root / "data" / "api_batch_results.json"
    run_eval_batch(out)
