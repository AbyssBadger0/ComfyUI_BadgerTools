from PIL import Image
import numpy as np

# 计算两个颜色之间的距离
def color_distance(color1, color2):
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

# 寻找主要颜色
def find_dominant_color(block, threshold):
    colors_count = {}
    for color in block:
        found_similar = False
        for dominant_color in colors_count:
            if color_distance(color, dominant_color) < threshold:
                colors_count[dominant_color] += 1
                found_similar = True
                break
        if not found_similar:
            colors_count[tuple(color)] = 1
    # 找出出现最多的颜色
    dominant_color = max(colors_count, key=colors_count.get)
    return np.mean([color for color in block if color_distance(color, dominant_color) < threshold], axis=0)

# 加载颜色卡
def load_color_card(color_card_image):
    color_card_pixels = np.array(color_card_image)
    color_palette = color_card_pixels.reshape(-1, color_card_pixels.shape[-1])
    return color_palette

# 将颜色匹配到颜色卡中的颜色
def match_color_to_palette(color, palette):
    closest_colors = sorted(palette, key=lambda c: color_distance(c, color))
    return tuple(closest_colors[0])

# 规格化原图片
def regular_image(original_image, output_size=(512, 512), fill_color=(255, 255, 255)):

    # 计算等比缩放的尺寸
    original_width, original_height = original_image.size
    ratio = min(output_size[0] / original_width, output_size[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 等比缩放图片
    try:
        original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        original_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)

    # 创建一个新的白色背景图片
    new_img = Image.new("RGB", output_size, fill_color)

    # 计算居中位置
    left = (output_size[0] - new_width) // 2
    top = (output_size[1] - new_height) // 2

    # 将缩放后的图片粘贴到白色背景图片上
    new_img.paste(original_image, (left, top))

    original_pixels = np.array(new_img)
    return original_pixels