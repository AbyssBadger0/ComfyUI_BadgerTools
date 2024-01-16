import math
import os
import uuid
from PIL import Image
import numpy as np
import torch
import comfy.utils
from .videoCut import getCutList, video_to_frames, cutToDir, frames_to_video
from .seg import get_masks
from .thick_lines_from_canny import fill_white_segments, find_largest_white_component


def getImageSize(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    return size


def tensorToImg(imageTensor):
    imaget = imageTensor[0]
    i = 255. * imaget.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


def imgToTensor(img):
    image = np.array(img).astype(np.float32) / 255.0
    imaget = torch.from_numpy(image)[None,]
    return imaget


def img_to_mask(mask):
    mask = mask.convert("RGBA")
    mask = np.array(mask.getchannel('R')).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)
    return mask


def img_to_np(img):
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = np.array(img)
    return img


def np_to_img(numpy):
    return Image.fromarray(numpy.astype(np.uint8))


def maskimg_to_mask(mask_img):
    mask = np_to_img(mask_img)
    mask = img_to_mask(mask)
    mask = mask.unsqueeze(0)
    return mask


class ImageOverlap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "additional_image": ("IMAGE",),
                "x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "overlap"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def overlap(self, base_image, additional_image, x, y):
        b_image = tensorToImg(base_image)
        a_image = tensorToImg(additional_image)

        b_image.paste(a_image, (x, y))
        o_image = imgToTensor(b_image)
        return (o_image,)


class FloatToInt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"})
            },
        }

    RETURN_TYPES = ("INT",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "floatToInt"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def floatToInt(self, float):
        return (round(float),)


class IntToString:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "intToString"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def intToString(self, int):
        return (str(int),)


class FloatToString:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.00001,
                    "round": False,
                    "display": "number"})

            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "floatToString"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def floatToString(self, float):
        return (str(float),)


class ImageNormalization:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "width": ("INT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "height": ("INT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "target_width": ("INT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "target_height": ("INT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"})
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("new_width", "new_height", "top", "left", "bottom", "right")

    FUNCTION = "imageNormalization"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def imageNormalization(self, width, height, target_width, target_height):
        o_ratio = width / height
        ratio = target_width / target_height
        top = 0
        left = 0
        bottom = 0
        right = 0
        nw = 0
        nh = 0
        # 原图比期望尺寸更扁，对齐宽，计算高，补上下
        if (o_ratio >= ratio):
            upratio = target_width / width
            nw = target_width
            nh = round(height * upratio)
            hdiff = target_height - nh
            top = math.floor(hdiff / 2)
            bottom = math.ceil(hdiff / 2)
        else:
            upratio = target_height / height
            nw = round(width * upratio)
            nh = target_height
            wdiff = target_width - nw
            left = math.floor(wdiff / 2)
            right = math.ceil(wdiff / 2)

        return (nw, nh, top, left, bottom, right,)


class ImageScaleToSide:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side_length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "side": (["Longest", "Shortest", "Width", "Height"],),
                "upscale_method": (cls.upscale_methods,),
                "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imageUpscaleToSide"

    CATEGORY = "badger"

    def imageUpscaleToSide(self, image, upscale_method, side_length: int, side: str, crop):
        samples = image.movedim(-1, 1)

        size = getImageSize(image)

        width_B = int(size[0])
        height_B = int(size[1])

        width = width_B
        height = height_B

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = height_B / width_B
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
                height = side_length
                width = width_ratio * height
            return width, height

        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        elif side == "Shortest":
            if width < height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)

        width = math.ceil(width)
        height = math.ceil(height)

        cls = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        cls = cls.movedim(1, -1)
        return (cls,)


class StringToFizz:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING", "INT",)

    FUNCTION = "stringToFizz"
    CATEGORY = "badger"

    def stringToFizz(self, text):
        textA = text.split("\n")
        lines = 0
        outText = ""
        for line in textA:
            if (len(line) > 0):
                line = "\"" + str(lines) + "\":\"" + line + "\",\n"
                lines = lines + 1
                outText = outText + line
        outText = outText[:-2]
        return (outText, lines,)


class TextListToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"texts": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)

    INPUT_IS_LIST = True

    FUNCTION = "textListToString"
    CATEGORY = "badger"

    def textListToString(self, texts):
        fullString = ""
        if len(texts) <= 1:
            return (texts,)
        else:
            for text in texts:
                fullString += text + "\n"
            return (fullString,)


class getImageSide:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side_choose": (["short", "long"],)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "getImageSide"

    CATEGORY = "badger"

    def getImageSide(self, image, side_choose):

        size = getImageSize(image)

        width = int(size[0])
        height = int(size[1])

        side = 0
        if width > height:
            if side_choose == "short":
                side = height
            else:
                side = width
        else:
            if side_choose == "short":
                side = width
            else:
                side = height

        return (side,)


class VideoToFrame:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": None}),
                "save_name": ("STRING", {"default": "temp"}),
                "min_side_length": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "frame_rate": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "video_to_frame"

    CATEGORY = "badger"

    def video_to_frame(self, video_path, save_name, min_side_length, frame_rate):
        videoPath = os.path.abspath(video_path)
        imagePath = video_to_frames(videoPath, min_side_length, frame_rate, save_name)

        return (imagePath,)


class VideoCutFromDir:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_dir": ("STRING", {"default": None}),
                "min_frame": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "max_frame": ("INT", {
                    "default": 240,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "video_cut_from_dir"

    CATEGORY = "badger"

    def video_cut_from_dir(self, frame_dir, min_frame, max_frame):
        cutList = getCutList(frame_dir, min_frame, max_frame)
        dirPathString = cutToDir(frame_dir, cutList)

        return (dirPathString,)


class FrameToVideo:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_dir": ("STRING", {"default": ""}),
                "save_path": ("STRING", {"default": "result.mp4"}),
                "frame_rate": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "frame_to_video"

    CATEGORY = "badger"

    def frame_to_video(self, frame_dir, save_path, frame_rate):
        save_path = os.path.abspath(save_path)
        frames_to_video(frame_dir, frame_rate, save_path)
        return (save_path,)


class getParentDir:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "getParentdir"

    CATEGORY = "badger"

    def getParentdir(self, dir_path):
        dir_path = os.path.abspath(dir_path)
        parent_path = os.path.dirname(dir_path)
        return (parent_path,)


class mkdir:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir_path": ("STRING", {"default": ""}),
                "new_dir": ("STRING", {"default": "newdir"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "mkdir"

    CATEGORY = "badger"

    def mkdir(self, dir_path, new_dir):
        dir_path = os.path.abspath(dir_path)
        new_dir_path = os.path.join(dir_path, new_dir)
        if not os.path.exists(new_dir_path):
            os.mkdir(new_dir_path)
        return (new_dir_path,)


class findCenterOfMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("X", "Y",)
    FUNCTION = "find_center_of_mask"

    def find_center_of_mask(self, mask):
        if mask.dim() == 3:
            mask = mask.squeeze(0)  # Remove the channel dimension if it exists
        assert mask.dim() == 2, "Mask must be 2D"

        # Create grids for x and y coordinates
        h, w = mask.size()
        x_coords = torch.arange(w).float().to(mask.device)
        y_coords = torch.arange(h).float().to(mask.device)

        # Compute the center of mass (centroid) of the mask
        total_mass = mask.sum()
        if total_mass > 0:
            x_center = (mask.sum(dim=0) * x_coords).sum() / total_mass
            y_center = (mask.sum(dim=1) * y_coords).sum() / total_mass
        else:
            x_center, y_center = torch.tensor(0), torch.tensor(0)

        # Convert to int
        X = float(x_center.item())
        Y = float(y_center.item())

        return (X, Y,)


class SegmentToMaskByPoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img": ("IMAGE",),
                "X": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "Y": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "dilate": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 4096.0,
                    "step": 1,
                    "display": "number"
                }),
                "sam_ckpt": ("SAM_MODEL",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("MASK", "MASK", "MASK",)
    RETURN_NAMES = ("mask0", "mask1", "mask2",)
    FUNCTION = "seg_to_mask_by_point"

    def seg_to_mask_by_point(self, img, X, Y, dilate, sam_ckpt):
        img = tensorToImg(img)
        img = img_to_np(img)
        latest_coords = [X, Y]
        masks = get_masks(img, latest_coords, dilate, sam_ckpt)
        mask0 = maskimg_to_mask(masks[0])
        mask1 = maskimg_to_mask(masks[1])
        mask2 = maskimg_to_mask(masks[2])

        return (mask0, mask1, mask2,)


class CropImageByMask:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("cropped_img", "X", "Y",)
    FUNCTION = "crop_image_by_mask"

    def crop_image_by_mask(self, image, mask):
        # Ensure the mask is binary
        mask = (mask > 0.5).float()

        # Find the bounding box of the mask
        if mask.sum() == 0:
            raise ValueError("The mask is empty, cannot determine bounding box for cropping.")

        # Find indices where the mask is nonzero
        nonzero_indices = torch.nonzero(mask.squeeze(0), as_tuple=True)
        topmost = torch.min(nonzero_indices[0])
        leftmost = torch.min(nonzero_indices[1])
        bottommost = torch.max(nonzero_indices[0])
        rightmost = torch.max(nonzero_indices[1])

        # Crop the image using the bounding box
        cropped_image = image[:, topmost:bottommost + 1, leftmost:rightmost + 1]

        # Return the cropped image and the top-left coordinates of the bounding box
        X = int(leftmost)
        Y = int(topmost)
        return (cropped_image, X, Y,)


class ApplyMaskToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)
    FUNCTION = "apply_mask_to_image"

    def apply_mask_to_image(self, image, mask):
        image = tensorToImg(image)
        mask = tensorToImg(mask)
        mask = mask.convert("L")

        # 将图片转换为RGBA，以便添加透明度通道
        image = image.convert("RGBA")

        # 分离图片的通道
        r, g, b, a = image.split()

        # 将蒙版应用为alpha通道
        new_a = Image.composite(a, Image.new('L', mask.size, 0), mask)

        # 合并图像通道和新的alpha通道
        result_image = Image.merge('RGBA', (r, g, b, new_a))
        return (imgToTensor(result_image),)


class DeleteDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("STRING", {"default": None}),
                "dir_path": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "badger"
    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "delete_dir"

    def delete_dir(self, start, dir_path):
        e_info = ""
        abs_dir_path = os.path.abspath(dir_path)
        if not os.path.exists(abs_dir_path):
            e_info = "路径不存在"
            return (0, e_info,)
        else:
            try:
                # 遍历文件夹中的每个文件或子文件夹
                for root, dirs, files in os.walk(abs_dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)  # 删除文件

                    for folder in dirs:
                        folder_path = os.path.join(root, folder)
                        os.rmdir(folder_path)  # 删除空文件夹

                os.rmdir(abs_dir_path)  # 最后删除根目录
                e_info = "成功删除"
                return (1, e_info,)
            except Exception as e:
                e_info = str(e)
                return (0, e_info,)


class FindThickLinesFromCanny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "find_thick_lines_from_canny"

    def find_thick_lines_from_canny(self, image, low_threshold, high_threshold):
        img = tensorToImg(image)
        result = fill_white_segments(img, low_threshold, high_threshold)
        result = find_largest_white_component(result)
        result = result.convert("RGB")
        result_tensor = imgToTensor(result)
        return (result_tensor,)


class TrimTransparentEdges:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim_transparent_edges"

    def trim_transparent_edges(self, image):
        img = tensorToImg(image)
        img = img.convert("RGBA")

        # 获取图片数据
        datas = img.getdata()

        # 获取非透明像素的边界
        non_transparent_pixels = [
            (i % img.width, i // img.width)
            for i, pix in enumerate(datas)
            if pix[3] != 0
        ]
        if not non_transparent_pixels:
            raise ValueError("Image is fully transparent")

        # 获取非透明像素的最小和最大坐标
        x_min = min(x for x, _ in non_transparent_pixels)
        y_min = min(y for _, y in non_transparent_pixels)
        x_max = max(x for x, _ in non_transparent_pixels)
        y_max = max(y for _, y in non_transparent_pixels)

        # 裁剪图片
        cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped_img = imgToTensor(cropped_img)

        return (cropped_img,)


class ExpandImageWithColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "bottom": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "left": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "right": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "color": ("STRING", {"default": None}),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "expand_image_with_color"

    def expand_image_with_color(self, image, top, bottom, left, right, color=None):
        img = tensorToImg(image)

        # Determine the new size of the image
        new_width = img.width + left + right
        new_height = img.height + top + bottom

        # Create a new image with the new size and the given background color
        if color:
            new_img = Image.new("RGBA", (new_width, new_height), color)
        else:
            # Use transparency if no color was provided
            new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Paste the original image onto the new image
        new_img.paste(img, (left, top), img)

        result = imgToTensor(new_img)

        return (result,)


class GetUUID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "append": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_uuid"

    def get_uuid(self, append, seed):
        result = uuid.uuid4().hex + append
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ImageOverlap-badger": ImageOverlap,
    "FloatToInt-badger": FloatToInt,
    "IntToString-badger": IntToString,
    "FloatToString-badger": FloatToString,
    "ImageNormalization-badger": ImageNormalization,
    "ImageScaleToSide-badger": ImageScaleToSide,
    "StringToFizz-badger": StringToFizz,
    "TextListToString-badger": TextListToString,
    "getImageSide-badger": getImageSide,
    "VideoCutFromDir-badger": VideoCutFromDir,
    "FrameToVideo-badger": FrameToVideo,
    "VideoToFrame-badger": VideoToFrame,
    "getParentDir-badger": getParentDir,
    "mkdir-badger": mkdir,
    "findCenterOfMask-badger": findCenterOfMask,
    "SegmentToMaskByPoint-badger": SegmentToMaskByPoint,
    "CropImageByMask-badger": CropImageByMask,
    "ApplyMaskToImage-badger": ApplyMaskToImage,
    "deleteDir-badger": DeleteDir,
    "FindThickLinesFromCanny-badger": FindThickLinesFromCanny,
    "TrimTransparentEdges-badger": TrimTransparentEdges,
    "ExpandImageWithColor-badger": ExpandImageWithColor,
    "GetUUID-badger": GetUUID

}

NODE_DISPLAY_NAME_MAPPINGS = {
}
