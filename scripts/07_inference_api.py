import sys
import json
import re
from PIL import Image # 图片处理（打开，转RGB，缩放）

from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_fix_dir = str(_project_root / "env" /  "lib" /  "python3.10" / "site_packages_fix")  # 修复conda env路径
if _fix_dir not in sys.path:
    sys.path.insert(0, _fix_dir)

import torch      # GPU推理
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig  # 加载模型，处理器，量化配置
from peft import PeftModel   # 挂载lora权重
# from pathlib import Path




# 默认prompt（和训练数据04,推理06保持一致）
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
        # 加载4bit量化模型 + LoRA权重（bf16+nf4, 和训练一般）
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
        # 单张图推理：图片预测 -> messages -> generate -> 切片 -> decode
        if prompt is None:
            prompt = DEFAULT_PROMPT
        
        image = image.convert("RGB")
        image.thumbnail((800, 800))   # 缩放到800px以内（3050 4GB显存限制）

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(   # 把对话格式转成模型能理解的文本
            messages, tokenize=False, add_generation_prompt=True
        )


        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)   # processor处理成tensor,放到GPU上


        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,         # 降为256（正常输出100-300字，512太大导致冗长列举）
                repetition_penalty=1.2,     # 惩罚重复，防止循环列举
            )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]  # 先切片，切掉的是输入部分的数字
            output_text = self.processor.batch_decode(   # # decode成人类可读文字，不decode的话是一串token id
                generated_ids, skip_special_tokens=True
            )[0]
        
            # 释放显存
            del inputs, generated_ids  # 删掉推理过程中间的两个大tensor（输入张量+输出张量），释放他们占的GPU内存
            torch.cuda.empty_cache()     # 告诉GPU把回收彻底清掉，还给系统
            # 不写，跑一张图没事，但是跑50张图时显存会一点点堆积，最终OOM（显存不够报错）

            # 后处理：切掉模型生成的垃圾尾巴（emoji、特殊符号、ASCII乱码等）
            output_text = self._clean_output(output_text)

            return output_text

    @staticmethod
    def _clean_output(text):
        """简单清理：匹配到明确垃圾标记就切掉，不过滤正常内容"""
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
        # 批量处理（复用06的逻辑）
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



        

















