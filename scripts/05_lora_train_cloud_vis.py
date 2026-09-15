import json
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

'''
peft = Parameter-Efficient Fine-Tuning(参数高效微调库)  ？？？？？？？？详细了解一下，在大模型很多地方都听过fine-tuning
LoraConfig = LoRA的配置类，高速模型“LoRA怎么加”
get_peft_model = 把普通模型包装成LoRA模型的函数

'''
from peft import LoraConfig, get_peft_model
# peft库在背后做了这些事：
# 1. 找到模型里所有 q_proj 和 v_proj 层
# 2. 给每个层【新建】两个空矩阵 A 和 B
# 3. A填入随机小数，B全填0
# 4. 把A、B挂到 q_proj/v_proj 旁边

# A和B不是从任何文件加载的，是peft库当场创建的
# 就像你写 x = [0,0,0] 一样，是新造出来的


from qwen_vl_utils import process_vision_info   # 处理图片的函数?  不懂？？？？？？？？？？？？？？？？？
# 从message列表把图片/视频路径提取出来，自动用PIL打开图片，返回PIL Image对象列表

# print("=" * 60)
# print("数据导入")
# print("=" * 60)
print("=" * 60 + "\n数据导入\n" + "=" * 60)

base = "/root/autodl-tmp"

with open(base + "data/training_data.json", "r") as f:
    raw_data = json.load(f)
print(f"加载训练数据： {len(raw_data)} 条")

print("=" * 60 + "\n4bit量化\n" + "=" * 60)

# 4bit量化配置，把模型从16bit压缩到4bit，显存从14Gb降到2Gb
quantization_config = BitsAndBytesConfig  (
    load_in_4bit=True,                        # 启用4bit量化
    bnb_4bit_compute_dtype=torch.bfloat16,    # 计算时用什么精度。存储是4bit，但计算时要还原成更高精度
    bnb_4bit_use_double_quant=True,           # 双重量化，把量化超参数本身也量化一次，再省一点显存（省0.4bit/参数）
    bnb_4bit_quant_type="nf4",    # normalfloat4，一种专门为神经网络权重分布设计的4bit编码方式，比普通均匀量化精度更高
)   # bnb_4bit_quant_type="nf4"  还是不理解？？？？？
# 为什么用bfloat16不用float16? 之前autodl第一次训练用的float16，loss全是NaN。bfloat16的指数位和float32一样（8位），不容易溢出

# 对于LLM权重： NF4误差明显 更小！！！！！！！！！

'''
Q:怎么理解NF4？
A:NF4是专门针对“大模型权重近似正态分布”设计的4位量化方案。
  同样只有16个量化等级，但NF4把更多等级分配给0附近，从而显著降低LLM权重量化误差
Q:为什么0附近权重特别多？
A:神经网络存在一种现象：大量参数贡献很小。在训练过程中，很多参数被 优化到：接近0的状态 
  因此出现：大量权重集中在0附近，少量权重集中在两侧（即：典型的高斯分布）
'''

'''
重难点：理解 -- LLM的参数（权重）到底是什么？为什么能被NF4压缩？

Q:什么是LoRA？什么是QLoRA?
A:由于eg，LoRA想法是：原模型不动，只新增A矩阵 B矩阵，所以训练：ΔW = A × B，最后W_new = W + ΔW ，所以：LoRA训练参数：几百万
eg：7B模型：70亿参数 x 2Byte（FP16）= 14GB，训练时还要梯度+优化器状态+激活值。通常40~60GB显存起步
QLoRA：（进一步升级）LoRA:原模型 FP16，还占14GB。QLoRA：把原模型先压成：NF4再做LoRA，所以：LoRA=FP16模型+LoRA，QLoRA=NF4模型+LoRA
LoRA
└── 一种微调方法

QLoRA
└── LoRA + 4bit量化

NF4
└── 一种4bit量化格式

LoRA
│
├── 原模型 FP16
│
└── 训练 LoRA 参数

QLoRA
│
├── 原模型 NF4
│
└── 训练 LoRA 参数

Q:FP16/BF16是什么？
A:本质：浮点数存储格式。这是啥？有一个均匀分布的叫什么来着

Q:什么是显存？
A:不是硬盘，我的SN7100属于SSD固态硬盘（存文件）。显存（VRAM）:GPU专属内存。例如：RTX5070,12GB显存
训练模型时：权重必须放显存

Q:为什么4bit只能表示16个数
A:bit=二进制数。4bit = 2^4 =16（1bit=0,1两个状态；2bit=2^2=4:00,01,10,11；3bit=8个状态）
因为1100只需要4bit，而精确存储，至少要16bit
Q:为什么只存索引？
A:eg：NF4量化表：

Q:FP4和NF4？
A:横轴：权重值，纵轴：出现次数。fp4量化点：均匀间隔，nf4量化点：越靠近0越密

Q:为什么权重会变成高斯？
A:训练过程：前向传播->计算损失->反向传播->得到梯度->更新参数
权重 = 初始值 + 大量随机微小变化（趋向高斯）

Q:真的有大量接近0的权重吗？
A:很多接贡献很小，eg：0.003和0.000，对最终输出影响非常接近。所以优化器会把很多权重压到：接近0
因为接近0的特别多，远离0的特别少。这就是：极大值极小值很少 的意思

Q:”nf4把更多等级分配给0附近“是什么意思？
A:transformer的权重绝大部分都堆积在0附近，因此不要把16个量化等级平均平均分布到整个区间，而应该把更多等级挤到0附近，
让最常见的权重拥有更高的精度。这样在同样4bit（16个编码）的前提下，整体量化误差最小

Q:量化前后的 几bit是何意味？
A:量化后参数量没变，变得是每个参数占位的位数
量化前：30亿个 × 16bit = 48 Gbit ≈ 6 GB
量化后：30亿个 × 4bit  = 12 Gbit ≈ 1.5 GB
存储空间缩小了4倍，但参数个数还是30亿
'''



print("正在加载模型...")
model_path = base + "models/Qwen2.5-VL-3B-Instruct"
'''
VL = Vision-Language（视觉+语言）
ForConditionalGeneration = 条件生成模型（给图片+文字， 生成文字）
'''
#  从本地目录加载预训练模型（不是从网上读，是从已经下好的文件读）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config, # 传入上面的4bit配置，加载时直接量化（加载完就是4bit的）
    torch_dtype=torch.bfloat16, # 模型中未量化的部分（如LayerNorm）用     bfloat16      ???????????????????????????????????
    # bfloat比float的指数位大 ，之前在autodl训练时溢出的原因（float放不下：反向传播算出的梯度，溢出成NaN）
    device_map="auto", # 自动分配gpu/cpu    不可以强制吗？训练模型是不是强制会更好一点？
)
'''可以强制
device_map="auto"       # 自动分配，放不下时自动放cpu
device_map={"": "cuda"} # 强制全放gpu，放不下直接报错
device_map={"": "cpu"}  # 强制全放cpu（极慢）
'''


processor = AutoProcessor.from_pretrained(model_path) # processor = 处理器，负责把“图片+文字”转成模型能懂的数字
# 包含两个东西：tokenizer = 文字 -> 数字id（车道线 -> 1234， 5678）    分词器
# image_processor = 图片 -> 像素张量（jpg -> 三维数组）              图像处理器
# 模型和processor必须用同一个model_path加载，否则文字编码对不上
print("模型加载完成")

# LoRA配置，决定怎么加低秩矩阵    ？？？？？？？？？？？ 听不懂啥意思？另外微调有很多方法吧？为啥用lora？
lora_config = LoraConfig(
    r=8,    # 秩（rank），LoRA矩阵的中间维度。
# 原始权重是4096x4096，LoRA插入两个矩阵：4096x8 和 8x4096
# r越大 -> 参数越多 -> 学习能力越强，但越容易过拟合。8是社区常用值      这块也不懂？举例子？或者去找个视频学习一下
    lora_alpha=16, # 缩放系数。LoRA的输出会乘以
    # alpha/r = 2。alpha是r的两倍是常用设置，控制LoRA更新幅度    也不理解
    # 给这两层加 lora
    target_modules=["q_proj", "v_proj"], # q_proj:查询投影，v_proj:值投影（transformer的注意力机制中的Q和V）
    # 也可以加k_proj、o_proj等，但q+v是最核心的
    lora_dropout=0.05, # 训练时随机丢弃5%的LoRA参数，防止过拟合（类似正则化）  所以啥是lora，是一堆参数？怎么丢弃，0.05是随机丢弃？
    bias="none", # 不训练偏置顶（偏置参数太少，不值得训）
    task_type="CAUSAL_LM", # 任务类型：因果语言模型（从左到右生成文字）
)
'''
1、全量微调：训练时要更新所有37.5亿参数，需要给每个参数 存“梯度”、“优化器状态”，总共要几十g显存
   LoRA微调：冻结原始37。5亿参数不动，只在旁边加少量的“小参数”训练，省显存，训练快，效果接近全量微调
2、神经网络权重里的“有效信息”远小于他的参数量，用两个小矩阵的乘积就能近似表达“微调要改的那部分”
3、为什么选 LoRA？其他方法对比?
全量微调：改所有参数
  ✗ 几十GB显存，普通人玩不起

Adapter：在层之间插入新的小网络
  ✗ 推理时多了额外计算层，变慢

Prefix-Tuning：在输入前加可训练的"虚拟token"
  ✗ 占用输入长度，效果不稳定

LoRA：在权重旁加两个小矩阵
  ✓ 训练时只改小矩阵，省显存
  ✓ 训练完后可以把 B×A 合并回 W，推理时零额外开销
  ✓ 效果接近全量微调

QLoRA = 4bit量化 + LoRA
  ✓ 这就是你项目用的，最省显存的方案
4、参数太多，容易过拟合，此项目是“看驾驶场景图说话”，属于中等难度任务，r=8是社区经验，大多数lora微调都用这个
5、lora_alpha=16 — 控制LoRA的"话语权"
实际公式：输出 = W×输入 + (lora_alpha/r) × B×A×输入
                              ↑
                    这就是缩放系数 = 16/8 = 2

含义：LoRA学到的修正值要乘以2，再加到原始输出上

为什么需要这个系数？
  - LoRA矩阵A、B是随机初始化的，初期数值很小
  - 不放大就被原始W淹没，学不到东西
  - 放大2倍让LoRA的"话语权"和原始权重在同一量级

为什么 alpha=16，r=8（2倍关系）？
  - 这是经验值，alpha/r=2 在大多数任务上效果好
  - 比例太小（如alpha=2, r=8 → 0.25）：LoRA学的东西被淹没
  - 比例太大（如alpha=64, r=8 → 8）：训练不稳定，原始能力被破坏
  
6、target_modules=["q_proj", "v_proj"] — 给哪些层加LoRA
Transformer的注意力机制有4个核心线性层：

  q_proj（Query查询）   → "我要找什么"
  k_proj（Key键）       → "我有什么信息"
  v_proj（Value值）     → "具体内容是什么"
  o_proj（Output输出）  → "整合输出"

为什么选 q_proj + v_proj？
  - Q和V是最核心的：Q决定"关注哪里"，V决定"提取什么"
  - 微调这两个层性价比最高
  - 全加（q+k+v+o）参数翻倍，但效果提升不明显

实际算账：
  你模型有36层Transformer，每层都有q_proj和v_proj
  → 共72个位置插入LoRA矩阵
  → 总参数 ≈ 72 × 32,768 ≈ 235万（接近你看到的184万）
  
7、lora_dropout=0.05 — 随机丢弃5%
LoRA就是两个矩阵A和B，里面装着一堆浮点数参数，和普通神经网络参数一样。
dropout的工作方式：
  训练时，每次前向传播，A的输出（2048→8的中间结果）会被随机遮5%

  假设A输出是8个数：[0.3, 0.8, 0.1, 0.5, 0.7, 0.2, 0.9, 0.4]
  dropout=0.05 随机选5%（约0.4个位置）变成0
  
  实际操作：生成一个随机mask
  mask = [True, True, True, True, True, True, True, False]  
  （大约5%的位置是False）
  
  output = output * mask / (1 - 0.95)
  （剩下的数字放大补偿，保持总量不变）

为什么要丢弃？
  防止模型过度依赖某几个神经元
  → 每次丢失的位置不同
  → 模型被迫让所有神经元都学到有用信息
  → 推理时dropout关闭，所有神经元参与，效果更稳

0.05 = 5%，是很小的值，因为LoRA本身参数就少，丢太多会学不够

8、bias="none" — 不训练偏置
线性层公式：y = Wx + b
                    ↑
                  bias偏置，每个输出维度1个

比如 q_proj: [2048, 2048]
  W有2048×2048=4,194,304个权重参数
  b只有2048个偏置参数（占比0.05%）

偏置参数太少，训练它的收益可以忽略，所以设"none"省事

9、task_type="CAUSAL_LM" — 任务类型
CAUSAL_LM = Causal Language Model（因果语言模型）

"因果"：模型只能看到当前及之前的内容，不能偷看后面
  输入：[今天, 天气, 真]
  预测：[好]  ← 只能用前面的词预测下一个

这是GPT/Qwen这类"生成式模型"的标准设置
区别于 BERT 那种"完形填空"模型（能看前后文）

你的任务：看图 → 生成文字答案，属于CAUSAL_LM
'''


model = get_peft_model(model, lora_config) # 把原始模型包裹一层，在指定的q_proj和v_proj旁边插入LoRA那两个小矩阵矩阵
# 原始权重冻结不动（requires_grad=false）,只训练LoRA那两个小矩阵


'''
问题1：
LoRA的目的就是为了省显存？
核心是省显存，但不止这一点。 完整的目的：
1. 省显存（最直接）
   全量微调：要给37.5亿参数都算梯度+存优化器状态 → 几十GB
   LoRA：只给184万小参数算梯度 → 几MB
   → 普通人也能微调大模型

2. 训练快
   要更新的参数少了500倍 → 反向传播快得多
   你项目15分钟训完就是这个原因

3. 不破坏原模型
   原始W冻结不动 → 模型原有能力完全保留
   只在旁边加"补丁"→ 避免"灾难性遗忘"

4. 可以保存/切换多个任务
   原始模型7GB存一份
   任务A的LoRA：7MB（驾驶场景）
   任务B的LoRA：7MB（医疗问答）
   切换任务：换不同LoRA，不用换整个模型
关键洞察：LoRA发现了一个事实——"微调不需要改所有参数，改一小部分就能学到新任务"。
这是它的核心贡献，省显存只是这个发现带来的好处。

总结：
A和B哪来的？
  → peft库调用get_peft_model()时当场新建的，不是从文件加载的

A的初始值：随机小数（正态分布采样）
B的初始值：全0

为什么要这么设？
  → B=0 保证训练开始时LoRA不影响模型
  → A随机 保证有梯度可以传（如果A也=0，梯度也是0，学不动）
  → 训练过程中靠梯度下降逐渐调整A和B的值
类比：A是"画笔"（随机准备好），B是"颜料"（一开始没涂，是0），训练过程就是"逐渐上色"。
'''

###########     至此： QLoRA完成： nf4  +  LoRA   ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

# 打印参数统计
model.print_trainable_parameters()  # 这儿也不懂？总参数37.5亿，只训练184万（0.05%），省了99.95%的显存和计算

model.enable_input_require_grads() # 让输入需要梯度
'''
背景：4bit量化的模型有个问题——
  默认情况下，量化层的"输入"不计算梯度
  导致反向传播传到量化层就断了，LoRA的A、B收不到梯度

这行的作用：
  告诉模型"你的输入也要参与梯度计算"
  → 梯度能正常传过量化层
  → LoRA的A、B才能收到梯度并更新

不加这行的后果：训练时loss不下降（因为A、B收不到梯度）
'''

model.gradient_checkpointing_enable() # 用时间换显存
'''
正常训练：
  前向传播时，每层的中间结果（激活值）都存到显存
  反向传播时直接用
  → 快，但显存占用大

开启梯度检查点：
  前向传播时，不存中间结果
  反向传播需要时，重新算一遍前向
  → 慢30%，但省60-70%显存

为什么要开？
  VLM的图片激活值特别大（一张图切几百个patch，每个patch都存）
  不开的话32GB可能不够，开了就稳了

代价：训练时间从10分钟→15分钟，可以接受
'''

model.config.use_cache = False # 关闭KV缓存
'''
1、KV缓存是推理加速技术：
  生成第2个token时，把第1个token的K、V存起来
  生成第3个token时，不用重新算前2个token的K、V
  
  推理时：开启KV缓存 → 快很多
  
但训练时必须关闭，因为：
  1. 训练是"整条序列一起算"，不是逐token生成，用不上缓存
  2. KV缓存和梯度检查点冲突（一个要存，一个要不存）

所以：训练时关闭，推理时再开启

2、K和V是什么
来自Transformer的注意力机制：
输入 "红色汽车在行驶"

对每个词算3个向量：
  Q（Query 查询）：我要找什么样的信息？
  K（Key 键）：我能提供什么样的信息？
  V（Value 值）：我具体的内容是什么？

Q×K → 算出"哪些词和当前词相关"
  ↓
用相关性权重 × V → 得到当前词的新表示

3、梯度检查点的策略：前向传播时不存中间结果，反向时重算
KV缓存的策略：    前向传播时存K、V，后面复用

一个说"别存"，一个说"存下来"
两者同时开启 → 矛盾 → 会报错或行为异常

训练时：用梯度检查点（省显存更重要）→ 关闭KV缓存
推理时：用KV缓存（速度更重要）→ 不需要梯度检查点
'''

model.train()
'''
PyTorch模型有两种模式：
  model.train()  → 训练模式
  model.eval()   → 推理模式

区别：
  训练模式：Dropout开启（随机丢弃5%神经元）
           BatchNorm用当前batch的统计量
  
  推理模式：Dropout关闭（所有神经元都参与）
           BatchNorm用历史统计量

你配置了 lora_dropout=0.05
  → 训练模式才会生效（随机丢弃5%）
  → 推理模式Dropout关闭，不丢弃

所以必须调model.train()，否则Dropout不工作'''

# 优化器配置
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # 什么是优化器？为啥选这个？
'''
1、什么是优化器？
训练的目标：让模型的预测越来越准（loss越来越小）
优化器 = 决定“每个参数该怎么调、调多少”的算法

训练循环：
    1、前向传播 -> 算出loss（当前有多不准）
    2、反向传播 -> 算出梯度（每个参数该往哪个方向调）
    3、优化器  -> 根据梯度实际更新参数（调多少）
        参数_new = 参数_old - 学习率 x 梯度（最简单的SGD）

2、为什么选 AdamW？
SGD（最简单）：
    参数 = 参数 - lr x 梯度
    -> 只看当前这一步的坡度，容易被噪声带偏
    
AdamW（SGD的升级版）：
    1、动量：记住之前的方向，不会因为一步噪声突然转向
    2、自适应学习率：每个参数根据自己的历史梯度调整步长
        -> 经常更新的参数：步长缩小（已经接近最优）
        -> 很少更新的参数：步长保持（还需要多走）
    3、weight decay：轻微把步长往0拉，防止参数膨胀
->收敛快，稳，几乎不用调参
->2019年以后的事实标准，深度学习训练默认选它
    
'''


num_epochs = 3          # 训练轮数：把500条鼠完整过3遍
accumulation_steps = 8  # 梯度累计步数：每8个样本的梯度累加和才更新一次参数
# 模拟大batch，但显存放不下大batch

global_step = 0         # 全局步数计数器：记录”处理了多少个样本“，从0开始
log_loss = 0.0          # loss累加器：用来算平均loss，从0.0开始

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad()  # 清零梯度。pytorch有个特点: 梯度默认累加，不会自动清零

    for i, item in enumerate(raw_data): # enumerate带序号遍历：i  item
        messages = item["messages"]

        image_path = None
        for msg in messages:  # 查看一下训练数据：
            '''
            training_data.json 里每条数据长这样：

{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "/data/samples/n015-001.jpg"},  ← 图片路径
      {"type": "text", "text": "请分析这张驾驶场景..."}           ← 这张图对应的问题
    ]},
    {"role": "assistant", "content": "1.车道线：... 2.车辆：3辆小汽车..."}  ← 这张图对应的答案
  ]
}

            '''
            if isinstance(msg.get("content"), list):  # 如果内容是列表
                for part in msg["content"]:           # 逐个检查每个部分
                    if part.get("type") == "image":   # 找到图片类型的地方
                        image_path = part["image"]    # 提取图片路径，覆盖掉None

        if image_path and not os.path.exists(image_path):
        # 如果找到了路径但文件不存在
            print(f" 跳过缺失图片： {image_path}")
            continue
'''
SFTTrainer（自动）：
  遍历数据 → 遇到一张坏图 → 报错 → 整个训练崩溃 → 从头再来

手动训练循环（本脚本）：
  for i, item in enumerate(raw_data):
      try:
          处理数据 + 训练
      except Exception as e:
          print(f"跳过错误样本 {i}: {e}")
          continue   ← 跳过坏图，继续训练下一条
手动循环可以用 try/except 跳过坏图片，不影响其他数据训练。这是脚本里的第159-161行做的事。
''' # YES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                            PASS


        # 数据处理（把图片+文件转成模型能吃的数字）
        try:
            # 1、把messages格式化成模型的额对话模板
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # 2、从mssages提取图片 （PIL Image对象）
            image_inputs, video_inputs = process_vision_info(messages)
            # 3、把文字+图片一起处理成模型输入
            inputs = processor(
                text=[text],          # 上面对话模版的文本
                images=image_inputs,  # 图片对象
                videos=video_inputs,  # 视频对象（本项目没用）
                padding=True,         # 补齐长度
                return_tensors="pt",  # 返回pytorch张量
            ).to(model.device)        # 移到gpu
'''
# 步骤1：文字 → 对话模板文本
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# 步骤2：messages → 提取图片对象
image_inputs, video_inputs = process_vision_info(messages)

# 步骤3：文字+图片 → 模型输入
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)
换不同的VLM模型（LLaVA、InternVL等），这三步写法几乎一样。这是HuggingFace的标准套路。
唯一可能变的是参数名（比如有些模型不需要传videos）。
'''



        # 构建labels（高速模型”正确答案“是什么）
            labels = inputs["input_ids"].clone() # copy输入序列，作为答案
'''                     自回归训练!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
为什么labels = input_ids？
自回归训练：模型看前面的词，预测下一个词

input_ids: [用户, 请分析, 这张图, 助手, 1.车道线, 两条, 2.车辆, 三辆]
labels:    [请分析, 这张图, 助手, 1.车道线, 两条, 2.车辆, 三辆, <结束>]
           ↑ 每个位置都是"下一个正确答案"

所以labels可以直接从input_ids复制，错位一位
'''
            labels[labels == processor.tokenizer.pad_token_id] = -100
'''
-100 是什么？
labels[labels == processor.tokenizer.pad_token_id] = -100
-100 是PyTorch的"忽略标记"

为什么要忽略某些位置？

input_ids: [<pad>, <pad>, 用户, 请分析, ..., 1.车道线, 两条]
                                       ↑                ↑
                                   不需要预测         要预测（算loss）

labels:    [-100, -100, ..., ..., -100, 1.车道线, 两条]
            ↑ 忽略                   ↑ 忽略    ↑ 算loss

-100 的位置不算loss，模型只在"助手回答"部分学习
不在padding和用户输入上浪费算力
'''
            if hasattr(inputs, "image_grid_thw"):
                image_mask = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image":
                                image_mask.append(True)
                if image_mask:
                    labels[:, :1] = -100 # 把第0个位置设为-100
# Qwen2.5-VL的输入序列第一个token是<|vision_start|>（图片开始标记）.这个token不是要预测的内容 → 设为-100忽略
'''
这是手动训练循环，没有用 DataLoader，所以没有 batch_size=32 这种显式设置。每次循环处理1条数据 = batch_size=1。
对比：用DataLoader的写法（其他脚本常见的）
# 其他脚本可能这样写：
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for batch in dataloader:      # 每次取出8条数据
    outputs = model(**batch)  # 8条一起喂给模型
这个脚本的等价关系
正常脚本：                     本脚本：
batch_size=8                   batch_size=1（隐式）
一次更新参数                   accumulation_steps=8（累积8次再更新）
            '''

            # 前向传播 + 计算loss ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            outputs = model(**inputs, labels=labels) # 前向传播
            # 1、前向传播：模型预测 每一个位置的下一个词
            # 2、自动算loss：预测 vs labels，用CrossEntropyLoss（交叉熵损失）

            loss = outputs.loss / accumulation_steps # loss除以8
            # 因为梯度是loss的导数，loss除以8 = 梯度也除以8
            # 8个”除以8的梯度“累加 = 1个完整梯度的平均值
            # 数学上等价于batch_size=8的平均值


            # NaN检测： not a number（无效数值，通常因为数值溢出）
            if torch.isnan(loss):
                print(f" 跳过NaN样本 {i}")
                optimizer.zero_grad() # 清零梯度，NaN会污染梯度
                continue # 跳过这一条，处理下一条
'''
NaN = Not a Number（无效数值，通常是因为数值溢出）

如果loss是NaN，反向传播会让梯度也变成NaN → 优化器用NaN梯度更新参数 → 所有参数变NaN → 雪崩

所以：检测到NaN → 立即清零梯度 → 跳过这条 → 保护其他样本
'''


            # 反向传播 + 记录loss ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            loss.backward() # 反向传播：自动算出每个可训练参数的梯度

            log_loss += loss.item() * accumulation_steps     # 累加loss（用于打印log）
            total_loss += loss.item() * accumulation_steps   # 累加loss（用于epoch统计）
            step_count += 1   # 这个epoch的有效样本数+1
            global_step += 1  # 全局部署+1
'''
loss.item() 是什么？
  loss是一个PyTorch张量（GPU上的一个数字）
  loss.item() 把它转成普通Python数字（float）
  
  为什么要 * accumulation_steps？
  因为前面 loss = outputs.loss / 8（缩小了8倍）
  这里乘回8，还原成真实loss值，用于记录和打印
'''

            # 梯度累计 + 更新参数（每8此执行一次） ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            if global_step % accumulation_steps == 0:  # 每8个样本
                optimizer.step()        # 根据累计的梯度更新参数
                optimizer.zero_grad()   # 清零梯度，准备下一轮累计

                if global_step % 10 == 0:   # 每10次参数更新（注意：这里是被包在%8里面的）----> 也是打印log
                    avg = log_loss / 10     # 算最近10此更新的平均loss
                    print(f" Step {global_step}, AVG Loss: {avg:.4f}")
                    log_loss = 0.0  # 清零 ，重新累加

            # 薄检查点（每100此更新执行一次 ）
            if global_step % 100 == 0:
                model.save_pretrained(base + f"models/lora_checkpoint_step{global_step}")
                print(f" 检查点已保存： step {global_step}")
'''
为什么每100步存一次？

训练可能崩溃（断电、OOM、坏图片等）
如果崩溃了，所有进度丢失 → 从头训练 → 浪费时间

检查点 = 训练到某个时刻的LoRA权重快照
崩溃后可以从最近的检查点恢复，不用从头来

保存的文件夹：
  models/lora_checkpoint_step100/   ← 第100步的权重
  models/lora_checkpoint_step200/   ← 第200步的权重
  ...

每个检查点约7MB（只存LoRA参数，不存整个模型）
'''

        except Exception as e:
            print(f" 跳过错误样本 {i}: {e}")
            continue # 跳过坏图片，继续训练下一条

'''
处理1条数据：
  ┌─────────────────────────┐
  │ 处理图片+文字 → inputs   │
  │ 构建labels              │
  │ 前向传播 → 算loss       │
  │ loss ÷ 8               │
  │ NaN? → 跳过             │
  │ 反向传播（梯度累积）     │
  │ 记录loss               │
  │ global_step +1          │
  └─────────────────────────┘
           ↓
  每8次 → 更新参数 + 清零梯度
           ↓
  每10次更新 → 打印平均loss
  每100次更新 → 保存检查点
  
  出错 → 跳过，继续下一条
'''


    if step_count > 0:
        print(f"Epoch {epoch+1} 完成 | 平均Loss： {total_loss/step_count:.4f} | 有效样本： {step_count}")

model.save_pretrained(base + "models/lora_checkpoint")
print(f"\n训练完成！模型已保存到： {base}models/lora_checkpoint")


'''
loss = 模型预测有多不准
梯度 = 每个参数该怎么调才能减小loss

loss = outputs.loss
  ↓ 内部用的是 CrossEntropyLoss（交叉熵损失）
  ↓ 公式：loss = -Σ (label × log(预测概率))
  ↓ 你不用管，PyTorch自动算

loss.backward()
  ↓ 内部用的是 自动求导（autograd）
  ↓ 沿着计算图从loss往回算每个参数的偏导数
  ↓ 链式法则一层层传
  ↓ 你不用管，PyTorch自动算

optimizer.step()
  ↓ 内部：参数 = 参数 - lr × 梯度（AdamW还加了动量和自适应）
  ↓ 你不用管，PyTorch自动更新
'''



















