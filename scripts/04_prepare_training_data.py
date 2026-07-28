# 把nuscences的标注数据转成VLM微调用的训练数据

import json

name_cn = {
    "vehicle.car": "小汽车",
    "vehicle.truck": "卡车",
    "vehicle.bus.rigid": "公交车",
    "vehicle.motorcycle": "摩托车",
    "vehicle.bicycle": "自行车",
    "human.pedestrian.adult": "行人",
    "movable_object.trafficcone": "交通锥",
}

base = "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data/v1.0-trainval/"

with open(base + "category.json") as f1:
    categories = json.load(f1)
with open(base + "sample_data.json") as f2:
    sample_data = json.load(f2)
with open(base + "sample_annotation.json") as f3:
    annotations = json.load(f3)
with open(base + "instance.json") as f4:
    instance = json.load(f4)

cam_front_keys = []

for sd in sample_data:
    if sd["is_key_frame"] == True and "CAM_FRONT__" in sd["filename"]:
        cam_front_keys.append(sd)

count = len(cam_front_keys)
step = count // 500
selected = cam_front_keys[::step][:500]

inst_to_cat = {}
for inst in instance:
    inst_to_cat[inst["token"]] = inst["category_token"]

cat_to_name = {}
for cat in categories:
    cat_to_name[cat["token"]] = cat["name"]

ann_index = {}
for ann in annotations:
    token = ann["sample_token"]
    if token not in ann_index:
        ann_index[token] = []
    ann_index[token].append(ann)

training_data = []
for img in selected:
    st = img["sample_token"]
    found = ann_index.get(st, [])

    count_dict = {}
    for ann in found:
        cat_token = inst_to_cat[ann["instance_token"]]
        name = cat_to_name[cat_token]
        if name not in count_dict:
            count_dict[name] = 0
        count_dict[name] += 1

    vehicle_names = {"vehicle.car", "vehicle.truck", "vehicle.bus.rigid", "vehicle.motorcycle", "vehicle.bicycle"}
    pedestrian_names = {"human.pedestrian.adult"}
    sign_names = {"movable_object.trafficcone"}

    vehicles = []
    for en_name, num in count_dict.items():
        if en_name in vehicle_names:
            cn = name_cn.get(en_name, en_name)
            vehicles.append(f"{num}辆{cn}")

    pedestrians = []
    for en_name, num in count_dict.items():
        if en_name in pedestrian_names:
            cn = name_cn.get(en_name, en_name)
            pedestrians.append(f"{num}个{cn}")

    cones = []
    for en_name, num in count_dict.items():
        if en_name in sign_names:
            cn = name_cn.get(en_name, en_name)
            cones.append(f"{cn}")

    vehicle_text = "、".join(vehicles) if vehicles else "前方无可见车辆"
    pedestrian_text = "、".join(pedestrians) if pedestrians else "无行人"
    sign_text = "、".join(cones) if cones else "无交通标志或信号灯"

    text = (
        f"1. 车道线：根据图片判断车道线数量和类型。\n"
        f"2. 车辆：{vehicle_text}。{pedestrian_text}。\n"
        f"3. 交通标志/信号灯：{sign_text}。\n"
        f"4. 潜在驾驶风险：根据场景判断潜在风险。"
    )
    print(text)

    sample = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data/" + img["filename"]},
                {"type": "text", "text": "请分析这张驾驶场景图片，输出以下信息：\n1. 车道线数量和类型\n2. 前方车辆位置和距离估计\n3. 交通标志/信号灯状态\n4. 潜在驾驶风险"}
            ]},
            {"role": "assistant", "content": text}
        ]
    }
    training_data.append(sample)

with open("/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data//training_data.json", "w") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)
