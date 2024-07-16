import math
from PIL import Image

def rotate_image_with_padding(img:Image):
    # 创建一个空画布，
    canvas_size = (img.width * 3, img.height)
    canvas = Image.new('RGB', canvas_size, color='white')
    
    # 将原图片复制为三张，横着放着一行，居中放置在画布上
    for i in range(3):
        canvas.paste(img, (img.width * i + (canvas.width - img.width * 3) // 2, (canvas.height - img.height) // 2))

    # 将图片旋转45度，空白的地方用白色填上
    canvas = canvas.rotate(315, expand=True, fillcolor='white')
    point_x = (5*math.sqrt(2)-2)*img.width/4
    point_y = (3*math.sqrt(2)-2)*img.width/4
    croped_img = canvas.crop((point_x, point_y, point_x+img.width, point_y+img.height))
    return croped_img
