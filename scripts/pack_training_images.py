import os
import sys

base = "/media/work/Additional3/Agent/opencode/基于VLM的智能驾驶场景结构化理解系统"
src_dir = os.path.join(base, "data/samples/CAM_FRONT")
dst_dir = "/tmp/cam_front_500"
os.makedirs(dst_dir, exist_ok=True)

with open(os.path.join(base, "data/train_img_list.txt")) as f:
    names = [line.strip() for line in f if line.strip()]

ok = 0
fail = 0
for name in names:
    src = os.path.join(src_dir, name)
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        ok += 1
        continue
    try:
        with open(src, "rb") as fin:
            data = fin.read()
        with open(dst, "wb") as fout:
            fout.write(data)
        ok += 1
        if ok % 50 == 0:
            print(f"已复制 {ok} 张，失败 {fail} 张")
    except Exception as e:
        fail += 1
        print(f"跳过坏道文件: {name} ({e})")

print(f"\n完成: 成功 {ok}, 失败 {fail}")
if fail > 0:
    print(f"警告: {fail} 张图片因坏道无法读取，训练时会跳过这些样本")
