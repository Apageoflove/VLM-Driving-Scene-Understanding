# 把nuscences的标注数据转成VLM微调用的训练数据

import json

'''
1、加载3个json
2、筛选CAM_FRONT关键帧，取500张
3、用sample_token关联标注，统计每张图的物体类别和数量
4、生成文字描述，组装成VLM对话格式
5、保存为JSON
'''

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
# 路径末尾要有斜杠 /
with open(base + "category.json") as f1:
    categories = json.load(f1)
with open(base + "sample_data.json") as f2:
    sample_data = json.load(f2)
with open(base + "sample_annotation.json") as f3:
    annotations = json.load(f3)
with open(base + "instance.json") as f4:
    instance = json.load(f4)

# image_path = "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data/samples"
# for filename in os.listdir(image_path):
#     if filename.endswith((".jpg", ".png", ".jepg")):
# 直接从json里筛选，不触碰图片目录        

# print(""is_key_frame: true"" in sample_data) sample_data是列表，不是字典，不能用in来查
'''
            列表用for遍历，字典用[]取字段
'''

cam_front_keys = [] # 用列表
# count = len(cam_front_keys)

for sd in sample_data:
    # if sd["is_key_frame"] == True and "CAM_FRONT" in sd["filename"]:
    if sd["is_key_frame"] == True and "CAM_FRONT__" in sd["filename"]:
        cam_front_keys.append(sd)
count = len(cam_front_keys)
step = count // 500
selected = cam_front_keys[::step][:500] # 从头到尾每隔step取一个，[:500]的意思是：从0开始取只取500张

'''
for img in selected:
    st = img["sample_token"]
    for ann in annotations:
        if ann["sample_token"] == st
            print("找到物体") # 性能问题：500张图片 x 116万条标注 = 5。8亿次比较，太慢了
'''

inst_to_cat = {}
for inst in instance:
    inst_to_cat[inst["token"]] = inst["category_token"]
# print(f"{inst_to_cat}") 不要放到循环里，不然无限打印了！！！，这两个字典存的是对应关系
cat_to_name = {}
for cat in categories:
    cat_to_name[cat["token"]] = cat["name"]
# print(f"{cat_to_name}")
ann_index = {}
# 先把 annotations 按 sample_token 建一个字典索引
for ann in annotations: # 取ann里的sample_token
    token = ann["sample_token"]
    if token not in ann_index:
        ann_index[token] = []
    ann_index[token].append(ann)
# count_dict = {} # 统计数量 ： 不能放在这，不然所有图片的统计混在一起。

training_data = []
for img in selected: # 遍历500张图片
    st = img["sample_token"] # 拿到图片的sample_token
    found = ann_index.get(st, []) # found来自sample_annotation.json,提取的是：ann里的标注信息。
    '''
    ann_index.get(st, [])是字典的.get()方法：
    st - 要查的key
    [] - 如果没找到，返回空列表
    eg：查到3个物体就返回 [ann1, ann2, ann3],一张图没有标注就返回[](不是报错)
    '''

# inst_to_cat = {}
# for inst in instance:
#     inst_to_cat[inst["token"]] = inst["category_token"]
# cat_to_name = {}
# for cat in categories:
#     cat_to_name[cat["token"]] = cat["name"]

    count_dict = {} # 每张图重新统计一次
    for ann in found:
        cat_token = inst_to_cat[ann["instance_token"]]
        name = cat_to_name[cat_token]
        if name not in count_dict:
            count_dict[name] = 0
        count_dict[name] += 1
    # print(count_dict)

    # --- 旧版：只输出物体数量 ---
    # parts = []
    # for en_name, num in count_dict.items():
    #     cn = name_cn.get(en_name, "")
    #     if cn != "":
    #         parts.append(f"{num}个{cn}")
    # text = ",".join(parts)
    # print(text)
    # sample = {
    #     "messages": [
    #         {"role": "user", "content": [
    #             {"type": "image", "image": "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data/" + img["filename"]},
    #             {"type": "text", "text": "请描述这张驾驶场景图片中的车辆、行人和关键物体。"}
    #         ]},
    #         {"role": "assistant", "content": text}
    #     ]
    # }

    # --- 新版：四项格式，和推理prompt一致 ---
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
    } # 不唯一，但必须是：Qwen2.5-VL微调用的格式，这个格式是HuggingFace/trl库的标准对话格式，Qwen2.5-VL微调用的结构：
    '''
    - messages 列表，包含role为user和assistant的对话
    - user的content是数组（图片+文本）
    - assistant的content是文本回复
    如果想改，能改的是里面的内容（提问方式、描述风格），不是结构，结构是框架固定的，改了微调代码就读不了了
    '''
    training_data.append(sample)

with open("/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统/data//training_data.json", "w") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)


































    
    

