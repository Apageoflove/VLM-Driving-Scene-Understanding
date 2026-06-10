"""
04_extra_merge_pseudo_labels.py — 用 Baseline pseudo-label 替换训练数据中的模板句
读取 training_data.json 和 baseline_train_results.json，
从 Baseline 结果中提取车道线和风险两项替换模板句，车辆和标志保留不动，
保存为 training_data_round3.json
"""

import json
import re
from pathlib import Path

data_root = Path("/media/devin/Cru_P3Plus/基于VLM的智能驾驶场景结构化理解系统/data")

training_data_file = data_root / "training_data.json"
baseline_results_file = data_root / "baseline_train_results.json"

output_file = data_root / "training_data_round3.json"


def extract_item(text, item_num):
    """从Baseline输出中提取第N项的内容"""
    pattern = rf"{item_num}\.\s*[^：]+：(.+?)(?=\n\d\.|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def build_new_answer(original_answer, baseline_output):
    """用Baseline的车道线+风险替换原答案中的模板句，保留车辆和标志"""
    lane_line = extract_item(baseline_output, 1)
    risk = extract_item(baseline_output, 4)

    vehicle = extract_item(original_answer, 2)
    sign = extract_item(original_answer, 3)

    if not lane_line:
        lane_line = "根据图片判断车道线数量和类型。"
    if not risk:
        risk = "根据场景判断潜在风险。"

    new_answer = f"1. 车道线：{lane_line}\n2. 车辆：{vehicle}\n3. 交通标志/信号灯：{sign}\n4. 潜在驾驶风险：{risk}"
    return new_answer


def get_image_filename(image_path):
    """从训练数据的图片路径中提取文件名"""
    return Path(image_path).name


def main():
    print(f"读取训练数据: {training_data_file}")
    with open(training_data_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    print(f"  共 {len(training_data)} 条")

    print(f"读取Baseline推理结果: {baseline_results_file}")
    with open(baseline_results_file, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
    print(f"  共 {len(baseline_results)} 条")

    matched = 0
    unmatched = 0
    new_training_data = []

    for sample in training_data:
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

        if filename in baseline_results:
            baseline_output = baseline_results[filename]
            new_answer = build_new_answer(original_answer, baseline_output)
            matched += 1

            new_sample = {
                "messages": [
                    sample["messages"][0],
                    {"role": "assistant", "content": new_answer}
                ]
            }
            new_training_data.append(new_sample)
        else:
            print(f"  未匹配: {filename}")
            new_training_data.append(sample)
            unmatched += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_training_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！")
    print(f"  匹配并替换: {matched} 条")
    print(f"  未匹配（保留原数据）: {unmatched} 条")
    print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    main()
