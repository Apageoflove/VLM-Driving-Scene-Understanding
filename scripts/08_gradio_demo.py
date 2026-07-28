"""
08_gradio_demo.py — Gradio Web Demo
上传驾驶场景图片 → 输出四项结构化分析
用法：env/bin/python scripts/08_gradio_demo.py
然后浏览器打开 http://127.0.0.1:7860
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))

import importlib
module = importlib.import_module("07_inference_api")
DrivingSceneAnalyzer = module.DrivingSceneAnalyzer
from PIL import Image
import gradio as gr

model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
lora_path = project_root / "models" / "lora_checkpoint"

analyzer = DrivingSceneAnalyzer(model_path, lora_path)

eval_images_dir = project_root / "data" / "eval_images"


def predict(image):
    if image is None:
        return "请先上传一张图片"
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        result = analyzer.analyze(image)
        return result
    except Exception as e:
        return f"推理失败：{e}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 驾驶场景 VLM 结构化理解系统
        基于 **Qwen2.5-VL-3B-Instruct** + **QLoRA** 微调，上传驾驶场景图片，输出四项结构化分析：
        - 车道线数量和类型
        - 前方车辆位置和距离估计
        - 交通标志/信号灯状态
        - 潜在驾驶风险
        """
    )

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
            output_text = gr.Textbox(label="结构化分析结果", lines=18)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text)
    image_input.change(fn=predict, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
