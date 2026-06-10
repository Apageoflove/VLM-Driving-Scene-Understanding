"""
06_lora_eval_debug.py — 带实时输出监控的评估脚本
每张图推理完立即打印输出内容（前300字），方便定位问题
"""
import sys
from pathlib import Path

data_root = Path("/media/devin/Cru_P3Plus/基于VLM的智能驾驶场景结构化理解系统")
model_path = data_root / "models" / "Qwen2.5-VL-3B-Instruct"
image_folder = data_root / "data" / "eval_images"
output_file = data_root / "data" / "lora_eval_results_debug.json"
lora_path = data_root / "models" / "lora_round3"

import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("正在加载模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_path),
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(str(model_path))
print("模型加载完成")

model = PeftModel.from_pretrained(model, str(lora_path))
print("LoRA挂载完成")

prompt = (
    "请分析这张驾驶场景图片，输出以下信息：\n"
    "1. 车道线数量和类型\n"
    "2. 前方车辆位置和距离估计\n"
    "3. 交通标志/信号灯状态\n"
    "4. 潜在驾驶风险"
)

image_paths = sorted(
    p for ext in ("*.jpg", "*.png", "*.jpeg") for p in image_folder.glob(ext)
)
print(f"共找到 {len(image_paths)} 张评估图片")
print("=" * 80)

results = {}
for i, filepath in enumerate(image_paths, 1):
    print(f"\n[{i}/{len(image_paths)}] {filepath.name}")
    try:
        image = Image.open(filepath).convert("RGB")
        image.thumbnail((800, 800))

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
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                repetition_penalty=1.5,
            )
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        actual_token_count = new_tokens.shape[1]

        output_text = processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        results[filepath.name] = output_text

        char_count = len(output_text)
        status = "OK" if char_count < 400 else "LONG" if char_count < 800 else "LOOP"
        print(f"  [{status}] {char_count}字 / {actual_token_count} tokens")
        print(f"  前300字: {output_text[:300]}")
        if char_count > 400:
            print(f"  尾部200字: ...{output_text[-200:]}")

    except Exception as e:
        print(f"  失败: {e}")
        results[filepath.name] = f"[错误] {e}"

    finally:
        if "inputs" in dir():
            del inputs
        if "generated_ids" in dir():
            del generated_ids
        if "new_tokens" in dir():
            del new_tokens
        torch.cuda.empty_cache()

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 80}")
print(f"完成！共处理 {len(results)} 张图片")
print(f"结果已保存到: {output_file}")
