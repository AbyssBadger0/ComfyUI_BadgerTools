from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def to_pixel(original_image, threshold, pix, tile_size, color_card=None):
    regular_size = tile_size*pix
    original_pixels = regular_image(original_image,output_size=(regular_size,regular_size))
    # 创建新的图像，用于存储像素化的结果
    pixelated_image = Image.new('RGB', (pix, pix))
    if color_card!=None:
        color_palette = load_color_card(color_card)
        # 遍历每个8x8的方块
        for i in range(0, regular_size, tile_size):
            for j in range(0, regular_size, tile_size):
                # 获取当前方块
                block = original_pixels[i:i+tile_size, j:j+tile_size].reshape(-1, 3)
                # 找到主要颜色
                dominant_color = find_dominant_color(block, threshold)
                # 匹配到颜色卡中的颜色
                matched_color = match_color_to_palette(dominant_color, color_palette)
                # 将匹配的颜色赋给对应的像素点
                pixelated_image.putpixel((j // tile_size, i // tile_size), matched_color)
    else:
        # 遍历每个8x8的方块
        for i in range(0, regular_size, tile_size):
            for j in range(0, regular_size, tile_size):
                # 获取当前方块
                block = original_pixels[i:i+tile_size, j:j+tile_size].reshape(-1, 3)
                # 找到主要颜色
                dominant_color = find_dominant_color(block, threshold)
                # 将主要颜色的平均值赋给对应的像素点
                pixelated_image.putpixel((j // tile_size, i // tile_size), tuple(dominant_color.astype(int)))
    return pixelated_image

def map_colors_to_palette(image, palette):

    # 将图像和调色板转换为PyTorch张量
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).cuda()
    palette_tensor = torch.tensor(np.array(palette), dtype=torch.float32).cuda()

    # 重塑张量以便进行计算
    image_reshaped = image_tensor.view(-1, 3)
    palette_reshaped = palette_tensor.view(-1, 3)

    # 计算图像中每个像素与调色板中每种颜色的欧几里得距离
    distances = torch.cdist(image_reshaped, palette_reshaped)

    # 找到每个像素的最近颜色
    nearest_color_indices = torch.argmin(distances, dim=1)

    # 使用最近的颜色索引创建新图像
    new_image_flat = palette_reshaped[nearest_color_indices]
    new_image = new_image_flat.view(image_tensor.shape)

    # 将结果转换回PIL图像并保存
    result_image = Image.fromarray(new_image.byte().cpu().numpy())
    return result_image

def reduce_colors(img, n_colors=16):
    # 将图片转换为numpy数组
    img_array = np.array(img)
    
    # 将图片reshape为二维数组
    original_shape = img_array.shape
    pixels = img_array.reshape((-1, 3))
    
    # 使用K-means算法来减少颜色
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # 替换每个像素的颜色为其最近的中心点
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # 将新的像素值reshape回原来的形状
    new_img_array = new_pixels.reshape(original_shape).astype(np.uint8)
    
    # 创建新的图片
    new_img = Image.fromarray(new_img_array)
    
    return new_img

def convert_image_to_tensor(img):
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)
    return img_pt


def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)
    return Image.fromarray(result_rgb_np)

class PixelEffectModule(nn.Module):
    def __init__(self):
        super(PixelEffectModule, self).__init__()

    def create_mask_by_idx(self, idx_z, max_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z])
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def forward(self, rgb, param_num_bins, param_kernel_size, param_pixel_size):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256. * param_num_bins
        intensity_idx = intensity_idx.long()

        intensity = self.create_mask_by_idx(intensity_idx, max_z=param_num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0)

        r, g, b = r * intensity, g * intensity, b * intensity

        kernel_conv = torch.ones([param_num_bins, 1, param_kernel_size, param_kernel_size])
        r = F.conv2d(input=r, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        g = F.conv2d(input=g, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        b = F.conv2d(input=b, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        intensity = F.conv2d(input=intensity, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins,
                             bias=None)[0, :, :, :]
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)


        r = torch.permute(r, dims=[1, 2, 0])
        g = torch.permute(g, dims=[1, 2, 0])
        b = torch.permute(b, dims=[1, 2, 0])

        r = self.select_by_idx(r, intensity_argmax)
        g = self.select_by_idx(g, intensity_argmax)
        b = self.select_by_idx(b, intensity_argmax)

        r = r / intensity_max
        g = g / intensity_max
        b = b / intensity_max

        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb_scale = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        return result_rgb,result_rgb_scale

class Photo2PixelModel(nn.Module):
    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()

    def forward(self, rgb,
                param_kernel_size=10,
                param_pixel_size=16):
        rgb,rgb_scale = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)
        return rgb,rgb_scale
    

def resize_image(image, max_size, is_pixel=False):
    # Image.LANCZOS,Image.NEAREST
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    if is_pixel:
        sample_type = Image.NEAREST
    else:
        sample_type = Image.LANCZOS
    resized_image = image.resize((new_width, new_height), sample_type)
    
    return resized_image

def convert_photo_to_pixel(img,abstraction,pixel_size,pixel_tile_size,preview_size):
    img = resize_image(img,pixel_size*pixel_tile_size)
    img_tensor = convert_image_to_tensor(img)
    model = Photo2PixelModel()
    model.eval()
    
    with torch.no_grad():
        rgb,rgb_scale = model(img_tensor,param_kernel_size = abstraction,param_pixel_size = pixel_tile_size)
    
    img_output = convert_tensor_to_image(rgb)
    img_preview = convert_tensor_to_image(rgb_scale)
    img_preview = resize_image(img_preview,preview_size,is_pixel=True)
    return img_output,img_preview