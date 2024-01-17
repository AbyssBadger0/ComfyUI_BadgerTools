from collections import defaultdict
import numpy as np
from PIL import Image


def rgb_to_hex(rgb_colr):
    return '{:02x}{:02x}{:02x}'.format(*rgb_colr)


def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_colors(PIL_img, n):
    color_list = []
    img = PIL_img.convert('RGBA')  # 确保图片是RGBA模式

    # 获取图片尺寸
    width, height = img.size

    for y in range(height):
        count = 0
        # 从左到右扫描
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            if a != 0:
                count += 1
                if count <= n:
                    color = (r, g, b)
                    color_list.append(rgb_to_hex(color))
            else:
                count = 0
        count = 0
        # 从右到左扫描
        for x in range(width - 1, -1, -1):
            r, g, b, a = img.getpixel((x, y))
            if a != 0:
                count += 1
                if count <= n:
                    color = (r, g, b)
                    color_list.append(rgb_to_hex(color))
            else:
                count = 0


    return color_list


def color_distance(c1, c2):
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def average_color(colors):
    r = int(np.mean([c[0] for c in colors]))
    g = int(np.mean([c[1] for c in colors]))
    b = int(np.mean([c[2] for c in colors]))
    return f"{r:02x}{g:02x}{b:02x}"


def fuzzy_color_grouping(colors, threshold):
    groups = defaultdict(list)

    for color in colors:
        rgb = hex_to_rgb(color)
        placed = False

        for group_color in groups:
            if color_distance(rgb, hex_to_rgb(group_color)) < threshold:
                groups[group_color].append(rgb)
                placed = True
                break

        if not placed:
            groups[color].append(rgb)

    return groups


def most_common_fuzzy_color(colors, threshold):
    groups = fuzzy_color_grouping(colors, threshold)
    largest_group = max(groups, key=lambda k: len(groups[k]))
    return average_color(groups[largest_group])


def is_color_similar(color1, color2, threshold):
    return all(abs(c1 - c2) <= threshold for c1, c2 in zip(color1, color2))


def find_similar_colors(image, color_string, threshold):
    # 转换颜色字符串为RGB元组
    target_color = tuple(int(color_string[i:i + 2], 16) for i in (0, 2, 4))

    # 创建一个同样大小的黑色背景图像
    output_image = Image.new('RGB', image.size, (0, 0, 0))
    pixels = image.load()
    output_pixels = output_image.load()

    # 遍历每个像素点，检查颜色是否接近目标颜色
    for x in range(image.width):
        for y in range(image.height):
            if is_color_similar(pixels[x, y], target_color, threshold):
                # 将接近的颜色设置为白色
                output_pixels[x, y] = (255, 255, 255)

    return output_image

