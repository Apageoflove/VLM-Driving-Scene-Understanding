"""
GPU使用通用模板 — VLM推理
用法：项目env/bin/python gpu_template.py
"""
import site
site.ENABLE_USER_SITE = False
import sys
_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if p != _user_site]

from pathlib import Path
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def main():
    project_root = Path(__file__).resolve().parent.parent.parent / "基于VLM的智能驾驶场景结构化理解系统"
    model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
    image_path = project_root / "data" / "eval_images" / "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915302412465.jpg"
    output_file = Path(__file__).resolve().parent / "output.json"

    # ========== GPU使用关键步骤 ==========

    # 第1步：检查GPU是否可用
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("GPU不可用，将使用CPU（速度会很慢）")

    # 第2步：配置量化（4bit，省显存）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 第3步：加载模型到GPU（device_map="auto"自动选择GPU）
    print("加载模型中...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    print("模型加载完成")

    # 第4步：准备输入数据
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((800, 800))

    prompt = "请分析这张驾驶场景图片，简要描述。"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 第5步：把数据送到GPU（.to(model.device)）
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    # 第6步：推理（torch.no_grad()省显存）
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # 第7步：只取模型新生成的部分（切片去掉输入）
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

    # 第8步：解码输出
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    print(f"\n模型输出:\n{output_text}")

    # 第9步：保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"output": output_text}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")

    # 第10步：释放GPU显存
    del inputs, generated_ids
    torch.cuda.empty_cache()
    print("GPU显存已释放")


if __name__ == "__main__":
    main()
