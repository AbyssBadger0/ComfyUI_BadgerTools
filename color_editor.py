from collections import defaultdict
import numpy as np
from PIL import Image


def rgb_to_hex(rgb_colr):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_colr)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def hex_to_rgba(hex_color, alpha=255):
    """
    将十六进制颜色字符串转换为RGBA格式。
    默认透明度（alpha）为255（不透明）。
    """
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + (alpha,)


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


def get_neighbors(x, y, width, height):
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < width - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < height - 1:
        neighbors.append((x, y + 1))
    return neighbors


def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def average_color(colors):
    r = int(np.mean([c[0] for c in colors]))
    g = int(np.mean([c[1] for c in colors]))
    b = int(np.mean([c[2] for c in colors]))
    return f"#{r:02x}{g:02x}{b:02x}"


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


def is_similar_color(target_color, current_color, threshold):
    return color_distance(target_color, current_color) <= threshold


def find_similar_colors(image, color_string, threshold):
    color_string = color_string.lstrip('#')
    # 转换颜色字符串为RGB元组
    target_color = tuple(int(color_string[i:i + 2], 16) for i in (0, 2, 4))

    # 创建一个同样大小的黑色背景图像
    output_image = Image.new('RGB', image.size, (0, 0, 0))
    pixels = image.load()
    output_pixels = output_image.load()

    # 遍历每个像素点，检查颜色是否接近目标颜色
    for x in range(image.width):
        for y in range(image.height):
            if is_similar_color(target_color,pixels[x, y],  threshold):
                # 将接近的颜色设置为白色
                output_pixels[x, y] = (255, 255, 255)

    return output_image


def detect_outline(image, target_hex_color, threshold):
    target_color = hex_to_rgb(target_hex_color)
    image = image.convert("RGBA")
    data = np.array(image)

    # Extract color and alpha channels
    color_data = data[:, :, :3]
    alpha_data = data[:, :, 3]

    # Create a mask for the outline
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    # Start from the edges of the image
    edge_pixels = [(x, y) for x in range(image.width) for y in [0, image.height - 1]] + \
                  [(x, y) for x in [0, image.width - 1] for y in range(1, image.height - 1)]

    # Use a queue to perform a breadth-first search from the edges
    queue = edge_pixels[:]
    while queue:
        x, y = queue.pop(0)
        if alpha_data[y, x] > 0 and is_similar_color(target_color, color_data[y, x], threshold) and mask[y, x] == 0:
            # Mark as part of the outline
            mask[y, x] = 1
            # Add neighbors to the queue
            for neighbor in get_neighbors(x, y, image.width, image.height):
                if mask[neighbor[1], neighbor[0]] == 0:
                    queue.append(neighbor)

    # Create a new image with a black background
    result_image = Image.new("RGB", (image.width, image.height), "black")
    result_data = np.array(result_image)

    # Draw the white pixels based on the mask
    result_data[mask == 1] = (255, 255, 255)

    # Convert back to PIL image
    result_image = Image.fromarray(result_data)
    return result_image
