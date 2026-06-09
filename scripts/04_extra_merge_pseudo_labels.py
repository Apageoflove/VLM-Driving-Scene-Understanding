"""
04_extra_merge_pseudo_labels.py — 用 Baseline pseudo-label 替换训练数据中的模板句
用法：/home/devin/conda_envs/vlm_drive/bin/python scripts/04_extra_merge_pseudo_labels.py

逻辑：
  1. 读取 04 生成的原 training_data.json（500条，车辆+标志有真实标注，车道线+风险是模板废话）
  2. 读取 Baseline 对500张训练图的推理结果 baseline_train_results.json
  3. 从 Baseline 结果中提取车道线和风险两项
  4. 替换 training_data.json 中的模板句，车辆和标志保留不动
  5. 保存为 training_data_round3.json
"""

import json
import re
from pathlib import Path

# 项目数据路径（外接硬盘）
data_root = Path("/media/devin/Cru_P3Plus/基于VLM的智能驾驶场景结构化理解系统/data")

# 输入文件
training_data_file = data_root / "training_data.json"                    # 04生成的原训练数据
baseline_results_file = data_root / "baseline_train_results.json"        # 新脚本跑的Baseline推理结果

# 输出文件
output_file = data_root / "training_data_round3.json"


def extract_item(text, item_num):
    """从Baseline输出中提取第N项的内容

    Baseline输出格式：
        1. 车道线数量和类型：两条车道，中间黄色实线。
        2. 前方车辆位置和距离估计：...
        3. 交通标志/信号灯状态：...
        4. 潜在驾驶风险：...
    """
    # 匹配 "数字. 标题：内容" 直到下一个 "数字." 或文本末尾
    pattern = rf"{item_num}\.\s*[^：]+：(.+?)(?=\n\d\.|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def build_new_answer(original_answer, baseline_output):
    """用Baseline的车道线+风险替换原答案中的模板句

    保留：车辆（第2项）、标志（第3项）
    替换：车道线（第1项）、风险（第4项）
    """
    # 从Baseline输出提取车道线和风险
    lane_line = extract_item(baseline_output, 1)   # 车道线
    risk = extract_item(baseline_output, 4)        # 风险

    # 从原答案提取车辆和标志（保留nuScenes真实标注）
    vehicle = extract_item(original_answer, 2)      # 车辆
    sign = extract_item(original_answer, 3)         # 标志

    # 如果Baseline提取失败，保留原答案（模板句也比没有强）
    if not lane_line:
        lane_line = "根据图片判断车道线数量和类型。"
    if not risk:
        risk = "根据场景判断潜在风险。"

    # 拼接新答案
    new_answer = f"1. 车道线：{lane_line}\n2. 车辆：{vehicle}\n3. 交通标志/信号灯：{sign}\n4. 潜在驾驶风险：{risk}"
    return new_answer


def get_image_filename(image_path):
    """从训练数据的图片路径中提取文件名

    原路径：/media/work/.../data/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__xxx.jpg
    提取：n015-2018-07-18-11-07-57+0800__CAM_FRONT__xxx.jpg
    """
    return Path(image_path).name


def main():
    # 1. 读取原训练数据
    print(f"读取训练数据: {training_data_file}")
    with open(training_data_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    print(f"  共 {len(training_data)} 条")

    # 2. 读取Baseline推理结果
    print(f"读取Baseline推理结果: {baseline_results_file}")
    with open(baseline_results_file, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    print(f"  共 {len(baseline_results)} 条")

    # 3. 建立文件名 → Baseline输出的映射
    # baseline_results 的key是文件名，value是Baseline的完整输出文本

    # 4. 逐条替换
    matched = 0
    unmatched = 0
    new_training_data = []

    for sample in training_data:
        # 从user的content里找到图片路径
        user_content = sample["messages"][0]["content"]
        image_path = None
        for item in user_content:
            if item.get("type") == "image":
                image_path = item["image"]
                break

        if not image_path:
            print(f"  警告：找不到图片路径，跳过")
            new_training_data.append(sample)
            unmatched += 1
            continue

        filename = get_image_filename(image_path)
        original_answer = sample["messages"][1]["content"]

        # 在Baseline结果里查找
        if filename in baseline_results:
            baseline_output = baseline_results[filename]
            new_answer = build_new_answer(original_answer, baseline_output)
            matched += 1

            # 替换assistant的content
            new_sample = {
                "messages": [
                    sample["messages"][0],  # user 不变
                    {"role": "assistant", "content": new_answer}
                ]
            }
            new_training_data.append(new_sample)
        else:
            # Baseline没有这张图的结果，保留原数据
            print(f"  未匹配: {filename}")
            new_training_data.append(sample)
            unmatched += 1

    # 5. 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_training_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！")
    print(f"  匹配并替换: {matched} 条")
    print(f"  未匹配（保留原数据）: {unmatched} 条")
    print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    main()
