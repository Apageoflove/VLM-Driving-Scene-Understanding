"""
08_gradio_demo.py — Gradio Web Demo（v0.2：结构化展示 + 自动评测页）

v1 的缺陷（本版本修复）：
- 结果只有一坨裸文本
- 模型在 import 时就加载：没 GPU 的机器连页面都打不开，评测更是没有入口

现在两个标签页：
- 场景分析：上传图片 → 结构化四项展示（Markdown 渲染），模型按需加载
- 自动评测：上传预测 JSON + 真值 JSON → 指标表格 + 完整 Markdown 报告，
  纯 CPU 也能用（评测不依赖模型）

用法：env/bin/python scripts/08_gradio_demo.py → http://127.0.0.1:7860
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from vlm_drive.evaluator import evaluate, metrics_table_rows, write_report
from vlm_drive.render import analysis_to_markdown

import gradio as gr

model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
lora_path = project_root / "models" / "lora_checkpoint"
eval_images_dir = project_root / "data" / "eval_images"

_analyzer = None  # 按需加载：评测页不碰模型，无 GPU 也能启动


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vlm_drive.analyzer import DrivingSceneAnalyzer

        _analyzer = DrivingSceneAnalyzer(
            model_path, lora_path, repetition_penalty=1.2  # v1 的防循环修复
        )
    return _analyzer


def predict(image):
    """场景分析页：图片 → 结构化记录 → Markdown 渲染。"""
    if image is None:
        return "请先上传一张图片"
    try:
        record = get_analyzer().analyze(image, image_name="upload.jpg")
        return analysis_to_markdown(record)
    except Exception as e:
        return f"推理失败：{e}"


def run_evaluation(pred_file, gt_file):
    """自动评测页：两个 JSON → 指标表格 + 完整报告（纯 CPU）。"""
    if not pred_file or not gt_file:
        return None, "请同时上传预测结果 JSON 和真值 JSON"
    try:
        predictions = json.loads(Path(pred_file).read_text(encoding="utf-8"))
        ground_truths = json.loads(Path(gt_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return None, f"文件读取失败：{e}"

    try:
        result = evaluate(predictions, ground_truths)
    except Exception as e:
        return None, f"评测失败：{e}"

    report_path = project_root / "data" / "demo_eval_report.md"
    write_report(result, report_path)
    return metrics_table_rows(result["metrics"]), (report_path.read_text(encoding="utf-8") + f"\n\n报告已保存到 `{report_path}`")


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 驾驶场景 VLM 结构化理解系统
        基于 **Qwen2.5-VL-3B-Instruct** + **QLoRA** 微调。
        """
    )

    with gr.Tab("场景分析"):
        gr.Markdown("上传驾驶场景图片，输出车道线 / 车辆 / 交通标志 / 驾驶风险四项结构化分析。模型首次推理时加载（约 30 秒）。")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传驾驶场景图片")
                submit_btn = gr.Button("开始分析", variant="primary")
                gr.Examples(
                    examples=[
                        str(eval_images_dir / "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915302412465.jpg"),
                        str(eval_images_dir / "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621804162404.jpg"),
                    ],
                    inputs=image_input,
                    label="示例图片（点击一键填入）",
                )
            with gr.Column():
                output_md = gr.Markdown(label="结构化分析结果")
        submit_btn.click(fn=predict, inputs=image_input, outputs=output_md)
        image_input.change(fn=predict, inputs=image_input, outputs=output_md)

    with gr.Tab("自动评测"):
        gr.Markdown(
            """
            上传预测结果与真值两个 JSON，自动计算结构化成功率、车辆/行人/交通锥命中率、
            按类别计数一致率等指标。纯 CPU 运行，不需要加载模型。

            预测 JSON 兼容两种格式：vlm_drive 的结构化记录，或 v1 脚本的裸文本输出。
            真值 JSON 用 `python -m vlm_drive gt` 从 nuScenes 标注生成。
            """
        )
        with gr.Row():
            pred_input = gr.File(label="预测结果 JSON（结构化记录或 v1 裸文本）")
            gt_input = gr.File(label="真值 JSON（vlm_drive gt 生成）")
        eval_btn = gr.Button("开始评测", variant="primary")
        metrics_table = gr.Dataframe(headers=["指标", "值"], label="评测指标", interactive=False)
        report_md = gr.Markdown()
        eval_btn.click(
            fn=run_evaluation,
            inputs=[pred_input, gt_input],
            outputs=[metrics_table, report_md],
        )

if __name__ == "__main__":
    demo.launch()
