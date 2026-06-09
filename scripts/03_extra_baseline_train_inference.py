"""
03_extra_baseline_train_inference.py — 用Baseline模型对500张训练图跑推理
用法：/home/devin/conda_envs/vlm_drive/bin/python scripts/03_extra_baseline_train_inference.py

目的：
  用 Qwen2.5-VL-3B（不挂LoRA）对500张训练图跑推理，
  获取每张图的车道线和风险描述，作为 pseudo-label 供 04_extra 拼接使用。

输出：data/baseline_train_results.json
  格式：{"图片文件名": "Baseline的四项分析文本", ...}
"""

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def main():
    # 路径配置
    data_root = Path("/media/devin/Cru_P3Plus/基于VLM的智能驾驶场景结构化理解系统")
    model_path = data_root / "models" / "Qwen2.5-VL-3B-Instruct"
    image_folder = data_root / "data" / "samples" / "CAM_FRONT"
    output_file = data_root / "data" / "baseline_train_results.json"

    # 量化配置（bf16+nf4，和训练一致，保证数值稳定）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载模型（不挂LoRA，纯Baseline）
    print("正在加载Baseline模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    print("模型加载完成")

    # 推理prompt（和训练/推理一致的四项格式）
    prompt = (
        "请分析这张驾驶场景图片，输出以下信息：\n"
        "1. 车道线数量和类型\n"
        "2. 前方车辆位置和距离估计\n"
        "3. 交通标志/信号灯状态\n"
        "4. 潜在驾驶风险"
    )

    # 从training_data.json读取500张训练图的文件名（04脚本选出的）
    training_data_file = data_root / "data" / "training_data.json"
    print(f"读取训练数据: {training_data_file}")
    with open(training_data_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    # 提取500张训练图的文件名
    target_filenames = set()
    for sample in training_data:
        for item in sample["messages"][0]["content"]:
            if item.get("type") == "image":
                target_filenames.add(Path(item["image"]).name)

    # 只扫描这500张图，不是全部34016张
    image_paths = sorted(
        p for ext in ("*.jpg", "*.png", "*.jpeg")
        for p in image_folder.glob(ext)
        if p.name in target_filenames
    )
    print(f"训练数据中 {len(target_filenames)} 张，在目录中找到 {len(image_paths)} 张")

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
                generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            # 切掉prompt部分，只保留模型回答
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

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共处理 {len(results)} 张图片")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
