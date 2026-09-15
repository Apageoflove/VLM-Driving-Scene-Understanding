"""
03_eval_inference.py — 对50张评估图片批量推理
用法：env/bin/python scripts/03_eval_inference.py
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_fix_dir = str(_project_root / "env" / "lib" / "python3.10" / "site_packages_fix")
if _fix_dir not in sys.path:
    sys.path.insert(0, _fix_dir)

import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
    image_folder = project_root / "data" / "eval_images"
    output_file = project_root / "data" / "eval_results.json"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("正在加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    print("模型加载完成")

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

    results = {}
    for i, filepath in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] 正在处理: {filepath.name}")

        try:
            image = Image.open(filepath).convert("RGB")
            max_size = 800
            image.thumbnail((max_size, max_size))

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
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            results[filepath.name] = output_text
            print(f"  完成，输出长度: {len(output_text)} 字")

        except Exception as e:
            print(f"  处理失败: {e}")
            results[filepath.name] = f"[错误] {e}"

        finally:
            if "inputs" in dir():
                del inputs
            if "generated_ids" in dir():
                del generated_ids
            torch.cuda.empty_cache()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共处理 {len(results)} 张图片")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
