"""Single CLI replacing the numbered inference/eval scripts.

Old entry points and what they map to:

- scripts/01_inference.py, scripts/03_eval_inference.py
      python -m vlm_drive infer --images data/eval_images --out data/eval_results.json
- scripts/06_lora_eval.py
      python -m vlm_drive infer --lora models/lora_round3 --out data/lora_eval_results.json
- (new) automatic scoring
      python -m vlm_drive evaluate --pred data/lora_eval_results.json --gt data/eval_gt.json --report data/report.md

The numbered scripts stay in the repo for training-pipeline reproducibility;
this CLI is the maintained path for everything downstream of a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_infer(args: argparse.Namespace) -> int:
    from .analyzer import DrivingSceneAnalyzer

    analyzer = DrivingSceneAnalyzer(
        args.model,
        lora_path=args.lora,
        output_format=args.format,
        repair_loop=args.repair,
    )
    analyzer.load_model()
    analyzer.analyze_folder(args.images, args.out, limit=args.limit)
    return 0


def _cmd_gt(args: argparse.Namespace) -> int:
    from .gt import build_gt

    gt = build_gt(args.nuscenes, args.images, min_visibility=args.min_visibility)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"真值生成：{len(gt)} 张 -> {out}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluator import evaluate_files

    result = evaluate_files(args.pred, args.gt, report_path=args.report)
    metrics = result["metrics"]
    print(f"评估图片: {metrics['images_evaluated']}  结构化成功率: {metrics['parse_success_rate']:.1%}")
    if args.report:
        print(f"报告: {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm_drive", description="VLM 驾驶场景结构化理解：推理与自动评估")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="批量推理（支持 base / LoRA，断点续跑）")
    infer.add_argument("--model", required=True, help="基座模型路径，如 models/Qwen2.5-VL-3B-Instruct")
    infer.add_argument("--lora", default=None, help="可选 LoRA 适配器路径，如 models/lora_round3")
    infer.add_argument("--images", required=True, help="图片目录")
    infer.add_argument("--out", required=True, help="输出 JSON 路径（已存在的条目自动跳过）")
    infer.add_argument("--format", choices=["text", "json"], default="text", help="输出格式：text=训练格式（默认，兼容 LoRA），json=严格 JSON")
    infer.add_argument("--repair", type=int, default=1, help="json 模式解析失败后的重试次数（默认 1）")
    infer.add_argument("--limit", type=int, default=None, help="只处理前 N 张（调试用）")
    infer.set_defaults(func=_cmd_infer)

    gt = sub.add_parser("gt", help="从本地 nuScenes 标注生成评估集真值（04 生成器格式）")
    gt.add_argument("--nuscenes", required=True, help="nuScenes 表目录，如 data/v1.0-trainval")
    gt.add_argument("--images", required=True, help="评估图片目录")
    gt.add_argument("--out", required=True, help="输出 GT JSON 路径")
    gt.add_argument("--min-visibility", type=int, default=1, choices=[1, 2, 3, 4],
                    help="nuScenes 可见度门槛 1-4（默认 1=不过滤；4=只保留 80-100% 可见物体，最贴近前摄视野）")
    gt.set_defaults(func=_cmd_gt)

    evaluate = sub.add_parser("evaluate", help="预测结果 vs nuScenes 标注的自动评分")
    evaluate.add_argument("--pred", required=True, help="预测结果 JSON（新版结构化记录或旧版裸文本均可）")
    evaluate.add_argument("--gt", required=True, help="标注 JSON：{图片名: 04 生成器格式文本}")
    evaluate.add_argument("--report", default=None, help="Markdown 报告输出路径（同时产出 .metrics.json）")
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
