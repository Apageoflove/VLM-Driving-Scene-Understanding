"""
00_resize_images.py — 图片预处理：批量缩小图片尺寸
作用：把 test_images/ 里的原图缩小到 800x800 以内，保存到 test_images_800/
原因：RTX 3050 只有 4GB 显存，原图（1600x900 或 3840x2160）太大，推理会爆显存
用法：env/bin/python scripts/00_resize_images.py
"""
import sys
import site
from pathlib import Path

site.ENABLE_USER_SITE = False  # 禁用用户级包，避免加载到旧版torch
_user_site = site.getusersitepackages()  # 获取用户级site-packages路径
sys.path = [p for p in sys.path if p != _user_site]  # 从搜索路径中移除它

from PIL import Image  # Pillow库，用来处理图片


def main():
    project_root = Path(__file__).resolve().parent.parent  # 从 scripts/ 往上一级 = 项目根目录
    input_folder = project_root / "data" / "test_images"  # 原始图片目录
    output_folder = project_root / "data" / "test_images_800"  # 缩小后的图片输出目录
    max_size = 800  # 最大边长（像素），长边缩到800，短边等比例缩

    output_folder.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）

    image_paths = sorted(  # 收集所有图片文件，按文件名排序
        p for ext in ("*.jpg", "*.png", "*.jpeg")  # 支持三种图片格式
        for p in input_folder.glob(ext)  # glob = 用通配符匹配文件
    )

    if not image_paths:  # 如果没找到任何图片
        print(f"错误：{input_folder} 中没有找到图片文件")
        return  # 直接退出函数

    print(f"找到 {len(image_paths)} 张图片，开始缩小到 {max_size}px ...")

    for filepath in image_paths:  # 遍历每张图片
        image = Image.open(filepath).convert("RGB")  # 打开图片，统一转RGB格式
        original_size = image.size  # 获取原始尺寸 (宽, 高)

        # ！！！！！！！！！！！！！！！
        image.thumbnail((max_size, max_size))  # 等比例缩小：长边不超过800，短边自动算
        # ！！！！！！！！！！！！！！！
        
        new_size = image.size  # 缩小后的尺寸

        output_path = output_folder / filepath.name  # 输出路径 = 输出目录 + 原文件名
        image.save(output_path, quality=90)  # 保存，quality=90 保证画质基本不变

        print(f"  {filepath.name}: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")

    print(f"\n完成！{len(image_paths)} 张图片已保存到: {output_folder}")


if __name__ == "__main__":  # 直接运行这个脚本才执行，被别的脚本import时不执行
    main()
