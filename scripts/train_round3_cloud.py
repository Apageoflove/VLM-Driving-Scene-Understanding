"""
Round 3 云端训练脚本（恒源云 3090 Ti）
上传到 /root/hy-tmp/ 直接运行：python /root/hy-tmp/train_round3_cloud.py
"""

import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

base = "/root/hy-tmp/"

# 加载训练数据
with open(base + "data/training_data_round3.json", "r") as f:
    raw_data = json.load(f)
print(f"加载训练数据: {len(raw_data)} 条")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("正在加载模型...")
model_path = base + "models/Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)
print("模型加载完成")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

num_epochs = 3
accumulation_steps = 8
global_step = 0
log_loss = 0.0

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad()

    for i, item in enumerate(raw_data):
        messages = item["messages"]

        # 自动替换图片路径为云端路径
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        filename = os.path.basename(part["image"])
                        part["image"] = base + "data/samples/CAM_FRONT/" + filename

        image_path = None
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        image_path = part["image"]

        if image_path and not os.path.exists(image_path):
            print(f"  跳过缺失图片: {os.path.basename(image_path)}")
            continue

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            if hasattr(inputs, "image_grid_thw"):
                image_mask = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image":
                                image_mask.append(True)
                if image_mask:
                    labels[:, :1] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps

            if torch.isnan(loss):
                print(f"  跳过NaN样本 {i}")
                optimizer.zero_grad()
                continue

            loss.backward()

            log_loss += loss.item() * accumulation_steps
            total_loss += loss.item() * accumulation_steps
            step_count += 1
            global_step += 1

            if global_step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if global_step % 10 == 0:
                    avg = log_loss / 10
                    print(f"  Step {global_step} | Loss: {avg:.4f}")
                    log_loss = 0.0

            if global_step % 100 == 0:
                model.save_pretrained(base + f"models/lora_round3_step{global_step}")
                print(f"  检查点已保存: step {global_step}")

        except Exception as e:
            print(f"  跳过错误样本 {i}: {e}")
            continue

    if step_count > 0:
        print(f"Epoch {epoch+1} 完成 | 平均Loss: {total_loss/step_count:.4f} | 有效样本: {step_count}")

model.save_pretrained(base + "models/lora_round3")
print(f"\n训练完成！LoRA权重已保存到 {base}models/lora_round3")
