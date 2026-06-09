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

# from 07_inference_api import DrivingSceneAnalyzer  # 报错：python文件名不能以数字开头直接import
import importlib
module = importlib.import_module("07_inference_api")  # 用importlib加载模块
DrivingSceneAnalyzer = module.DrivingSceneAnalyzer
from PIL import Image
import gradio as gr

model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
lora_path = project_root / "models" / "lora_checkpoint"           # LoRA权重

analyzer = DrivingSceneAnalyzer(model_path, lora_path)          # 加载（耗时约30s）

eval_images_dir = project_root / "data" / "eval_images"         # 示例图片目录，给gr.Examples用


def predict(image):                    # 连接gradio网页和07推理引擎的桥梁
    if image is None:
        return "请先上传一张图片"
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        result = analyzer.analyze(image)               # 调用07的推理方法
        return result              # 返回文字给网页
    except Exception as e:
        return f"推理失败：{e}"          # 出错也不崩网页


# gr.Blocks：比gr.Interface更灵活的布局方式，支持左右分栏、Markdown、自定义主题
with gr.Blocks(theme=gr.themes.Soft()) as demo:   # theme=Soft：柔和浅色主题，固定样式不随机
    gr.Markdown(                                   # gr.Markdown：在页面顶部插入Markdown格式的项目介绍
        """
        # 驾驶场景 VLM 结构化理解系统
        基于 **Qwen2.5-VL-3B-Instruct** + **QLoRA** 微调，上传驾驶场景图片，输出四项结构化分析：
        - 车道线数量和类型
        - 前方车辆位置和距离估计
        - 交通标志/信号灯状态
        - 潜在驾驶风险
        """
    )

    with gr.Row():                                 # gr.Row：左右两列布局（左=上传区，右=结果区）
        with gr.Column():                          # gr.Column：左列
            image_input = gr.Image(type="pil", label="上传驾驶场景图片")
            submit_btn = gr.Button("开始分析", variant="primary")   # variant="primary"：蓝色主按钮
            gr.Examples(                           # gr.Examples：示例图片区，点击一键填入，面试官不用自己找图
                examples=[
                    str(eval_images_dir / "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915302412465.jpg"),
                    str(eval_images_dir / "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621804162404.jpg"),
                ],
                inputs=image_input,                # 点击示例后自动填入image_input
                label="示例图片（点击一键填入）",
            )

        with gr.Column():                          # 右列
            output_text = gr.Textbox(label="结构化分析结果", lines=18)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_text)   # 点按钮触发推理
    image_input.change(fn=predict, inputs=image_input, outputs=output_text)  # 上传图片后也自动触发推理

if __name__ == "__main__":
    demo.launch()  # 启动本地Web服务器，默认 http://127.0.0.1:7860
