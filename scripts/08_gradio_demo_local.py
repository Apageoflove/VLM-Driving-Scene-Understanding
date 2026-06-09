"""
08_gradio_demo_local.py — Gradio Web Demo（本机5070ti适配版）
上传驾驶场景图片 → 输出四项结构化分析
用法：python scripts/08_gradio_demo_local.py
然后浏览器打开 http://127.0.0.1:7860
"""
import sys
from pathlib import Path

# 本机适配：数据在外接硬盘上
data_root = Path("/media/devin/Cru_P3Plus/基于VLM的智能驾驶场景结构化理解系统")
sys.path.insert(0, str(data_root / "scripts"))

import importlib
module = importlib.import_module("07_inference_api_local")
DrivingSceneAnalyzer = module.DrivingSceneAnalyzer
from PIL import Image
import gradio as gr

model_path = data_root / "models" / "Qwen2.5-VL-3B-Instruct"
lora_path = data_root / "models" / "lora_round3"

analyzer = DrivingSceneAnalyzer(model_path, lora_path)

eval_images_dir = data_root / "data" / "eval_images"


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


# 预定义示例图片路径
example_images = [
    str(eval_images_dir / "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915302412465.jpg"),
    str(eval_images_dir / "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621804162404.jpg"),
]

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

        with gr.Column():
            output_text = gr.Textbox(label="结构化分析结果", lines=18)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text)
    image_input.change(fn=predict, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
