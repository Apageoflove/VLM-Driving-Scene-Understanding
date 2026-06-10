import sys
import json
import re
from PIL import Image

from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_fix_dir = str(_project_root / "env" / "lib" / "python3.10" / "site_packages_fix")
if _fix_dir not in sys.path:
    sys.path.insert(0, _fix_dir)

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_PROMPT = (
    "请分析这张驾驶场景图片，输出以下信息：\n"
    "1. 车道线数量和类型\n"
    "2. 前方车辆位置和距离估计\n"
    "3. 交通标志/信号灯状态\n"
    "4. 潜在驾驶风险"
)


class DrivingSceneAnalyzer:
    """加载模型 + LoRA, 输入图片输出四项分析"""

    def __init__(self, model_path, lora_path):
        # bf16+nf4，和训练一致
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("正在加载模型...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        print("基座模型加载完成")

        print("正在挂载LoRA权重...")
        self.model = PeftModel.from_pretrained(self.model, str(lora_path))
        print("LoRA挂载完成")


    def analyze(self, image: Image.Image, prompt: str = None) -> str:
        if prompt is None:
            prompt = DEFAULT_PROMPT

        image = image.convert("RGB")
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                repetition_penalty=1.2,
            )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            del inputs, generated_ids
            torch.cuda.empty_cache()

            output_text = self._clean_output(output_text)

            return output_text

    @staticmethod
    def _clean_output(text):
        """匹配到明确垃圾标记就切掉，不过滤正常内容"""
        garbage_markers = [
            r'[\U0001F300-\U0001F9FF]',
            r'希望我的回答对你有所帮助',
            r'如果您有任何其他问题',
            r'请随时告诉我',
            r'祝您旅途愉快',
            r'欢迎随时提问',
            r'建议采取适当的预防措施',
            r'此外还可以考虑安装',
            r'对于此类复杂环境',
            r'我们应当更加注重',
            r'oOo',
            r'\\{3,}',
        ]
        cut_pos = len(text)
        for pat in garbage_markers:
            m = re.search(pat, text)
            if m and m.start() < cut_pos:
                cut_pos = m.start()
        if cut_pos < len(text):
            text = text[:cut_pos].rstrip()
            if text and text[-1] not in '。！？…；':
                text += '。'
        return text


    def analyze_batch(self, image_dir, output_file=None):
        image_dir = Path(image_dir)
        image_paths = sorted(
            p for ext in ("*.jpg", "*.png", "*.jpeg") for p in image_dir.glob(ext)
        )
        print(f"共找到 {len(image_paths)} 张图片")

        results = {}
        for i, filepath in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] {filepath.name}")
            try:
                image = Image.open(filepath)
                results[filepath.name] = self.analyze(image)
                print(f" 完成，{len(results[filepath.name])} 字")
            except Exception as e:
                print(f" 失败: {e}")
                results[filepath.name] = f"[错误] {e}"

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到： {output_file}")

        return results
