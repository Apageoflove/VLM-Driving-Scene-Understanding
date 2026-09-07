# 基于VLM的智能驾驶场景结构化理解系统

基于 **Qwen2.5-VL-3B-Instruct** + **QLoRA** 微调的驾驶场景结构化理解项目。输入一张驾驶场景图片，模型输出四项结构化分析：车道线、车辆、交通标志、驾驶风险。

## Demo 效果展示

<table>
  <tr>
    <td align="center"><b>城市道路停车</b></td>
    <td align="center"><b>路口场景</b></td>
    <td align="center"><b>雨天道路</b></td>
  </tr>
  <tr>
    <td><img src="assets/gradio_城市道路停车.jpg" width="400"/></td>
    <td><img src="assets/gradio_路口.jpg" width="400"/></td>
    <td><img src="assets/gradio_雨天道路.jpg" width="400"/></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><b>夜间道路 1</b></td>
    <td align="center"><b>夜间道路 2</b></td>
  </tr>
  <tr>
    <td><img src="assets/gradio_夜间道路1.jpg" width="400"/></td>
    <td><img src="assets/gradio_夜间道路2.jpg" width="400"/></td>
  </tr>
</table>

## 技术栈

| 类别 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-VL-3B-Instruct (7.1GB) |
| 微调方法 | QLoRA (4-bit NF4 + LoRA r=8) |
| 训练框架 | transformers + peft + trl |
| 量化配置 | BitsAndBytesConfig (bfloat16 + NF4) |
| 数据集 | nuScenes CAM_FRONT (500张训练 + 50张评估) |
| Web Demo | Gradio |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.6 |

## 环境安装

创建 conda 环境：

```bash
conda create -n vlm_drive python=3.10 -y
```

激活环境：

```bash
conda activate vlm_drive
```

安装 PyTorch（CUDA 12.6）：

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

安装项目依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

## 下载模型权重

安装 modelscope：

```bash
pip install modelscope
```

下载 Qwen2.5-VL-3B-Instruct 到本地：

```bash
modelscope download Qwen/Qwen2.5-VL-3B-Instruct --local_dir models/Qwen2.5-VL-3B-Instruct
```

## 数据准备

本项目使用 [nuScenes](https://www.nuscenes.org/nuscenes) 自动驾驶数据集，需前往官网注册下载。

生成训练数据（nuScenes 标注 → VLM 对话格式）：

```bash
python scripts/04_prepare_training_data.py
```

准备评估数据（从候选图中筛选 50 张）：

```bash
python scripts/02_prepare_eval_data.py
```

## 训练

本项目在云端 GPU（恒源云 RTX 3090 24GB）上完成 QLoRA 微调，训练约 36 分钟。

启动训练：

```bash
python scripts/05_lora_train_cloud.py
```

训练完成后 LoRA 权重保存在 `models/lora_round3/`

训练配置：

| 参数 | 值 |
|------|-----|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| 目标模块 | q_proj, v_proj |
| 训练轮数 | 3 epochs |
| 学习率 | 2e-4 |
| 梯度累积 | 8 steps |
| 量化 | 4-bit NF4 + bfloat16 |

## 推理

### Gradio Web Demo

启动 Demo：

```bash
python scripts/08_gradio_demo.py
```

浏览器打开 http://127.0.0.1:7860

### Python API 调用

单张图片推理：

```python
from scripts.07_inference_api import DrivingSceneAnalyzer
from PIL import Image

analyzer = DrivingSceneAnalyzer(
    model_path="models/Qwen2.5-VL-3B-Instruct",
    lora_path="models/lora_round3"
)

image = Image.open("demo_images/demo_城市道路停车.jpg")
result = analyzer.analyze(image)
print(result)
```

批量推理：

```python
results = analyzer.analyze_batch("data/eval_images/", output_file="results.json")
```

## 项目结构

```
├── scripts/
│   ├── 01_inference.py              # 单张图片推理
│   ├── 02_prepare_eval_data.py      # 评估数据准备
│   ├── 03_eval_inference.py         # Baseline 批量推理
│   ├── 04_prepare_training_data.py  # 训练数据生成（nuScenes → VLM 对话格式）
│   ├── 05_lora_train_cloud.py       # 云端 QLoRA 训练
│   ├── 06_lora_eval.py              # LoRA 模型评估推理
│   ├── 07_inference_api.py          # 推理模块封装（DrivingSceneAnalyzer 类）
│   └── 08_gradio_demo.py            # Gradio Web Demo
├── demo_images/                     # Demo 效果图（5张精选）
├── docs/
│   └── 问题总结/                    # 核心踩坑记录
├── requirements.txt                 # Python 依赖
└── README.md
```

## License

MIT

## v0.2：结构化输出管线与自动评估（vlm_drive 包）

> 大版本更新：项目从"脚本集合 + 人工打分"升级为"结构化管线 + 自动评估"。训练流程（04/05 编号脚本）保持不变，推理与评估全部迁移到 `vlm_drive` 包。

### 解决的三个核心问题

1. 输出不再是非结构化文本。旧管线把模型原文（甚至 `[错误] ...`）直接存进 JSON；现在每张图产出带 schema 校验的结构化记录（车道线 / 车辆计数 / 交通标志 / 风险），解析失败有一次自动修复重试，最终失败记为显式 error 记录。
2. 评估不再靠人工。`项目方案.md` 里的"四维度人工打分"由 evaluator 接管：结构化成功率、有无车辆/行人/交通锥命中率、按类别计数一致率与 MAE，一键产出 Markdown 报告（见 `examples/eval_demo/report.md`）。
3. 三份复制粘贴的推理循环合一。01/03/06 三个脚本的重复逻辑合并为 `DrivingSceneAnalyzer` 一个类，支持 base/LoRA、断点续跑、可注入生成函数（CPU 环境可测试）。

### 快速使用

```bash
# 推理（LoRA 权重，断点续跑——中断后重跑只处理剩余图片）
python -m vlm_drive infer --model models/Qwen2.5-VL-3B-Instruct --lora models/lora_round3 \
  --images data/eval_images --out data/lora_eval_results.json

# 评估集真值（从本地 nuScenes 标注生成，04 生成器格式；--min-visibility 过滤 360° 标注中严重遮挡的物体）
python -m vlm_drive gt --nuscenes data/v1.0-trainval --images data/eval_images --out data/eval_gt.json --min-visibility 4

# 自动评估（预测 vs nuScenes 标注，旧版裸文本输出也兼容）
python -m vlm_drive evaluate --pred data/lora_eval_results.json \
  --gt data/eval_gt.json --report data/report.md
```

### 测试

```bash
python -m unittest discover tests   # 31 项，纯 CPU、零 GPU 依赖
```

覆盖：训练格式解析（含编号变体/缺段）、JSON 修复（代码围栏/散文前缀/尾逗号/单引号/截断）、schema 校验、修复重试循环、生成失败降级、断点续跑、评估器各维度计分。
