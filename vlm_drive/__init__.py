"""vlm_drive — 结构化输出管线与自动评估，替代编号脚本的推理/评估循环。

Quick start:
    from vlm_drive.analyzer import DrivingSceneAnalyzer
    from vlm_drive.evaluator import evaluate_files

    # 推理（base 或 LoRA，断点续跑）
    analyzer = DrivingSceneAnalyzer("models/Qwen2.5-VL-3B-Instruct", lora_path="models/lora_round3")
    analyzer.load_model()
    analyzer.analyze_folder("data/eval_images", "data/lora_eval_results.json")

    # 自动评分（预测 vs nuScenes 标注）
    evaluate_files("data/lora_eval_results.json", "data/eval_gt.json", report_path="data/report.md")

CLI:
    python -m vlm_drive infer --model models/Qwen2.5-VL-3B-Instruct --images data/eval_images --out data/results.json
    python -m vlm_drive evaluate --pred data/results.json --gt data/eval_gt.json --report data/report.md
"""
__version__ = "0.1.0"
