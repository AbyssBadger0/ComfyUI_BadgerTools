from PIL import Image


def find_most_frequent_color(PIL_img, n):
    img = PIL_img.convert('RGBA')  # 确保图片是RGBA模式

    # 初始化颜色统计字典
    color_count = {}

    # 获取图片尺寸
    width, height = img.size

    for y in range(height):
        count = 0
        # 从左到右扫描
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            if a == 255:
                count += 1
                if count <= n:
                    color = (r, g, b)
                    color_count[color] = color_count.get(color, 0) + 1
            else:
                count = 0
        count = 0
        # 从右到左扫描
        for x in range(width - 1, -1, -1):
            r, g, b, a = img.getpixel((x, y))
            if a == 255:
                count += 1
                if count <= n:
                    color = (r, g, b)
                    color_count[color] = color_count.get(color, 0) + 1
            else:
                count = 0

    # 找到出现次数最多的颜色
    most_frequent_color = max(color_count, key=color_count.get)

    # 返回像素点最多的颜色对应的字符串（格式化为十六进制）
    return '#{:02x}{:02x}{:02x}'.format(*most_frequent_color)


def is_color_similar(color1, color2, threshold):
    return all(abs(c1 - c2) <= threshold for c1, c2 in zip(color1, color2))


def find_similar_colors(image, color_string, threshold):
    color_string = color_string[1:]
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
