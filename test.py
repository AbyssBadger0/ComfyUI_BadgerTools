import os
from PIL import Image

def split_image(image_path, rows, cols):
    """
    将一张图片均匀切分为rows行cols列，并保存到以原图片名加"_cut"为名的新文件夹中。
    
    :param image_path: 原始图片的路径
    :param rows: 切分的行数
    :param cols: 切分的列数
    """
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算每张小图的宽度和高度
    w, h = width // cols, height // rows
    
    # 创建保存切分后图片的文件夹
    folder_name = os.path.splitext(os.path.basename(image_path))[0] + "_cut"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 切分图片并保存
    for i in range(rows):
        for j in range(cols):
            box = (j * w, i * h, (j + 1) * w, (i + 1) * h)
            cropped_img = img.crop(box)
            cropped_img.save(f"{folder_name}/{os.path.splitext(os.path.basename(image_path))[0]}_{i*cols+j+1}.png")

# 使用示例
split_image("path_to_your_image.jpg", 3, 4)