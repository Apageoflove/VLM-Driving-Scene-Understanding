"""
02_prepare_eval_data.py — 从nuScenes全量图片中预筛选200张候选评估图片
从34149张CAM_FRONT图片里，按时间段均匀采样200张，分散在不同scene和日期
用户再从这200张里人工挑选50张覆盖不同场景
用法：env/bin/python scripts/02_prepare_eval_data.py
"""
import sys
import site
from pathlib import Path
import random
import shutil
import re
from collections import defaultdict

site.ENABLE_USER_SITE = False
_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if p != _user_site]


def parse_image_info(filename):
    """从nuScenes文件名中提取 scene编号, 日期, 小时"""
    match = re.match(r'(n\d+)-(\d{4})-(\d{2})-(\d{2})-(\d{2})-\d{2}-\d{2}', filename)
    if not match:
        return None, None, None
    scene = match.group(1)
    month = match.group(3)
    day = match.group(4)
    hour = match.group(5)
    date = f"{match.group(2)}-{month}-{day}"
    return scene, date, hour


def classify_time_period(hour):
    """把小时分成3个时段：上午(10-12)、下午(13-15)、傍晚(16-19)"""
    h = int(hour)
    if 10 <= h <= 12:
        return "上午"
    elif 13 <= h <= 15:
        return "下午"
    else:
        return "傍晚"


def main():
    random.seed(42)

    project_root = Path(__file__).resolve().parent.parent
    source_folder = project_root / "data" / "samples" / "CAM_FRONT"
    output_folder = project_root / "data" / "eval_candidates"
    num_candidates = 200

    output_folder.mkdir(parents=True, exist_ok=True)

    all_images = sorted(source_folder.glob("*.jpg"))
    print(f"源目录共 {len(all_images)} 张图片")

    groups = defaultdict(list)
    for img_path in all_images:
        scene, date, hour = parse_image_info(img_path.name)
        if scene is None:
            continue
        period = classify_time_period(hour)
        groups[(scene, date, period)].append(img_path)

    print(f"共 {len(groups)} 个分组（scene×日期×时段）")

    num_groups = len(groups)
    per_group = max(1, num_candidates // num_groups)
    print(f"每组约选 {per_group} 张")

    candidates = []
    for key, images in sorted(groups.items()):
        scene, date, period = key
        n = min(per_group, len(images))
        step = len(images) // n if n > 0 else 1
        sampled = [images[i * step] for i in range(n) if i * step < len(images)]
        candidates.extend(sampled)

    if len(candidates) < num_candidates:
        existing = set(candidates)
        remaining = [img for img in all_images if img not in existing]
        extra = random.sample(remaining, min(num_candidates - len(candidates), len(remaining)))
        candidates.extend(extra)

    if len(candidates) > num_candidates:
        candidates = random.sample(candidates, num_candidates)

    candidates.sort(key=lambda p: p.name)

    skipped = 0
    for img_path in candidates:
        try:
            shutil.copy2(img_path, output_folder / img_path.name)
        except OSError:
            skipped += 1

    period_count = defaultdict(int)
    scene_count = defaultdict(int)
    for img_path in candidates:
        scene, date, hour = parse_image_info(img_path.name)
        period = classify_time_period(hour)
        period_count[period] += 1
        scene_count[scene] += 1

    print(f"\n已选 {len(candidates)} 张候选图片，复制到: {output_folder}")
    if skipped > 0:
        print(f"（跳过 {skipped} 张因磁盘IO错误无法读取的图片）")
    print("\n时段分布：")
    for period in ["上午", "下午", "傍晚"]:
        print(f"  {period}: {period_count.get(period, 0)} 张")
    print("\nScene分布：")
    for scene, count in sorted(scene_count.items()):
        print(f"  {scene}: {count} 张")
    print("\n下一步：请打开 data/eval_candidates/ 浏览图片，手动挑选50张覆盖不同场景的图片")


if __name__ == "__main__":
    main()
