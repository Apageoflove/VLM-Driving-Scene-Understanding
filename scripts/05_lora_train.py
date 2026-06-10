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

import numpy
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

base = "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/"

with open(base + "data/training_data.json") as f:
    training_data = json.load(f)[:5]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("正在加载模型...")
model_path = base + "models/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_path),
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(str(model_path), min_pixels=128*28*28, max_pixels=256*28*28)
print("模型加载完成")

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 冻结视觉编码器，只训练语言模型的LoRA参数
for param in model.base_model.model.model.visual.parameters():
    param.requires_grad = False

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
num_epochs = 3

print(f"开始训练... {len(training_data)}条数据, {num_epochs}轮")

for epoch in range(num_epochs):
    total_loss = 0
    for i, sample in enumerate(training_data):
        messages = sample["messages"]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        from PIL import Image
                        image_inputs.append(Image.open(item["image"]))

        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        torch.cuda.empty_cache()

        print(f"  Epoch {epoch+1}/{num_epochs}, 样本 {i+1}/{len(training_data)}, loss={loss.item():.4f}")

    avg_loss = total_loss / len(training_data)
    print(f"Epoch {epoch+1} 完成, 平均loss={avg_loss:.4f}")

model.save_pretrained(base + "models/lora_checkpoint")
print("LoRA权重已保存！")


# ==================== 旧代码（SFTTrainer方式，OOM） ====================
#
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from datasets import load_dataset
#
# training_data = load_dataset("json", data_files=base + "data/training_data.json", split="train")
# training_data = training_data.select(range(5))
#
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_compute_dtype = torch.float16,
#     bnb_4bit_use_double_quant = True,
# )
# print("正在加载模型...")
# model_path = base + "models/Qwen2.5-VL-3B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     str(model_path),
#     quantization_config = quantization_config,
#     torch_dtype = torch.float16,
#     device_map = "auto",
# )
# processor = AutoProcessor.from_pretrained(str(model_path), min_pixels=128*28*28, max_pixels=256*28*28)
# print("模型加载完成")
#
# lora_config = LoraConfig(
#     r=4,
#     lora_alpha=8,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
#
# training_args = TrainingArguments(
#     output_dir=base + "models/lora_checkpoint",
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     save_steps=100,
#     logging_steps=10,
#     fp16=False,
#     bf16=True,
#     gradient_checkpointing=True,
# )
#
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=training_data,
#     processing_class=processor,
# )
#
# print("开始训练...")
# trainer.train()
# print("训练完成！")
#
# model.save_pretrained(base + "models/lora_checkpoint")
# print("LoRA权重已保存")
