"""
00_resize_images.py — 图片预处理：批量缩小图片尺寸
把 test_images/ 里的原图缩小到 800x800 以内，保存到 test_images_800/
用法：env/bin/python scripts/00_resize_images.py
"""
import sys
import site
from pathlib import Path

site.ENABLE_USER_SITE = False
_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if p != _user_site]

from PIL import Image


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_folder = project_root / "data" / "test_images"
    output_folder = project_root / "data" / "test_images_800"
    max_size = 800

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in ("*.jpg", "*.png", "*.jpeg")
        for p in input_folder.glob(ext)
    )

    if not image_paths:
        print(f"错误：{input_folder} 中没有找到图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片，开始缩小到 {max_size}px ...")

    for filepath in image_paths:
        image = Image.open(filepath).convert("RGB")
        original_size = image.size

        image.thumbnail((max_size, max_size))

        new_size = image.size

        output_path = output_folder / filepath.name
        image.save(output_path, quality=90)

        print(f"  {filepath.name}: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")

    print(f"\n完成！{len(image_paths)} 张图片已保存到: {output_folder}")


if __name__ == "__main__":
    main()
