"""
06_lora_eval.py — 对50张评估图片批量推理
用法：env/bin/python scripts/06_lora_eval.py
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

from peft import PeftModel

def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
    image_folder = project_root / "data" / "eval_images"
    output_file = project_root / "data" / "lora_eval_results.json"

    lora_path = project_root / "models" / "lora_checkpoint"

    # quantization_config = BitsAndBytesConfig(       # 旧版：fp16，和训练05的bf16不一致
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    quantization_config = BitsAndBytesConfig(          # 新版：bf16+nf4，和训练05保持一致
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # 推理时的量化配置应与训练时保持一致，
    # 否则模型在「训练看到的数值表示」和「推理实际的数值表示」之间存在分布偏移，削弱微调收益


    print("正在加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        # torch_dtype=torch.float16,        # 旧版：和训练不一致
        torch_dtype=torch.bfloat16,         # 新版：和训练05保持一致
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    print("模型加载完成")


    # 挂载lora权重
    model = PeftModel.from_pretrained(model, str(lora_path))
    # 基础模型不动，吧训练好的低秩适配器权重”贴“上去，推理时生效，这是PEFT推理的标准写法


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
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    repetition_penalty=1.2,   # 惩罚重复（Round3需1.5，Round2用1.2够了）
                )
                # generate返回的是：输入 + 新生成拼在一起的完整序列
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            '''
            切之前: [ p1, p2, ..., p10 | g1, g2, g3, g4, g5 ]
            切之后: [                    g1, g2, g3, g4, g5 ]   ← 只剩模型新生成的
            为什么要切？ 如果不切，下一步 decode 出来会把你输入的 prompt 原封不动也打印出来（「请分析这张驾驶场景图片，输出以下信息…」+ 模型的回答）。
            你要的是「模型的回答」，不是「prompt + 回答」。
            这是 VLM/LLM 推理的标准写法，几乎所有 Qwen/LLaMA 的推理脚本都这么切一刀。
            '''

            output_text = processor.batch_decode( # batch_decode永远返回列表
                generated_ids, skip_special_tokens=True
            )[0]
            # # 写法 B：用 decode（注意要取第 0 行）
            # output = processor.decode(generated_ids[0], skip_special_tokens=True)

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

# 切掉 prompt → id 转文字 → 去特殊符 → 存字典。其中第一行的切片是核心，是「只取模型回答」的关键操作。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!