import sys
import site
from pathlib import Path

site.ENABLE_USER_SITE = False
_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if p != _user_site]

_project_root = Path(__file__).resolve().parent.parent
_fix_dir = str(_project_root / "env" / "lib" / "python3.10" / "site_packages_fix")
if _fix_dir not in sys.path:
    sys.path.insert(0, _fix_dir)

import json

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def main():
    project_root = Path(__file__).resolve().parent.parent  # 定位项目根目录
    model_path = project_root / "models" / "Qwen2.5-VL-3B-Instruct"
    # image_folder = project_root / "data" / "test_images"
    image_folder = project_root / "data" / "eval_images"
    # output_file = project_root / "data" / "inference_results.json"
    output_file = project_root / "data" / "eval_results.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)  # 确保 data/ 存在·

    quantization_config = BitsAndBytesConfig(  # 4bit 量化，降低显存占用
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # 加载模型权重
        str(model_path),
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))  # 图片预处理 + tokenizer

    prompt = (
        "请分析这张驾驶场景图片，输出以下信息：\n"
        "1. 车道线数量和类型\n"
        "2. 前方车辆位置和距离估计\n"
        "3. 交通标志/信号灯状态\n"
        "4. 潜在驾驶风险"
    )

    image_paths = sorted(  # 收集所有图片，按文件名排序
        p for ext in ("*.jpg", "*.png", "*.jpeg") for p in image_folder.glob(ext)
    )
    if not image_paths:  # 没有图片就退出
        print(f"错误：{image_folder} 中没有找到图片文件")
        return

    results = {}
    # for filepath in image_paths[:10]:  # 跑全部10张，体验baseline效果
    for filepath in image_paths:
        print(f"正在处理: {filepath.name}")

        try:
            image = Image.open(filepath).convert("RGB")  # 统一转 RGB，避免 RGBA 报错
            
            # 防止4GB显存不够
            image.thumbnail((800, 800))
            
            messages = [  # 构造 Qwen2.5-VL 对话格式
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(  # 转换为模型训练时的对话格式
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(  # 文本和图片一起编码
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():  # 推理时不计算梯度，省显存
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]  # 只取模型新生成的部分

            output_text = processor.batch_decode(  # token → 文本
                generated_ids, skip_special_tokens=True
            )[0]
            results[filepath.name] = output_text
            print(f"  完成，输出长度: {len(output_text)} 字")

        except Exception as e:
            print(f"  处理失败: {e}")
            results[filepath.name] = f"[错误] {e}"

        finally:  # 每张图处理完释放显存
            del inputs, generated_ids
            torch.cuda.empty_cache()

    with open(output_file, "w", encoding="utf-8") as f:  # 保存结果 JSON
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共处理 {len(results)} 张图片")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":  # 直接运行才执行，被 import 不执行
    main()
