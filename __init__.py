import json
import math
import os
import hashlib
import uuid
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import requests
import torch
import comfy.utils
from .videoCut import getCutList, video_to_frames, cutToDir, frames_to_video
from .seg import get_masks
from .line_editor import fill_white_segments, find_largest_white_component
from .color_editor import get_colors, find_similar_colors, most_common_fuzzy_color, detect_outline,hex_to_rgba
from .image_editor import rotate_image_with_padding
from .pixel import *
import gc
import sys
import folder_paths


def getImageSize(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    return size


def maskTensorToImgTensor(maskTensor):
    return maskTensor.reshape((-1, 1, maskTensor.shape[-2], maskTensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)


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
    mask = mask.unsqueeze(0)
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
    return mask


def garbage_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class LoadImageAdvanced:
    def __init__(self):
        pass
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                "optional": 
                    {
                    "color": ("STRING", {"default": "#FFFFFF"}),
                    "upscale_method": (s.upscale_methods, {"default": "lanczos"}),
                    "target_width": ("INT", {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number"}),
                    "target_height": ("INT", {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number"}),
                    }
                }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_advanced"
    def load_image_advanced(self, image,color,upscale_method,target_width,target_height):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        width = img.size[0]
        height = img.size[1]
        nw = width
        nh = height
        top = 0
        left = 0
        bottom = 0
        right = 0
        if target_width > 0 and target_height > 0 and target_width != width and target_height != height:
            o_ratio = width / height
            ratio = target_width / target_height
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
            image = imgToTensor(img)
            samples = image.movedim(-1,1)
            s = comfy.utils.common_upscale(samples, nw, nh, upscale_method, crop="disabled")
            s = s.movedim(1,-1)
            img = tensorToImg(s)
        if color:
            rgba_color = hex_to_rgba(color)
            new_img = Image.new("RGBA",(nw+left+right,nh+top+bottom), rgba_color)
            new_img.paste(img, (left, top),img.convert("RGBA"))
            img = new_img

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB" if color else "RGBA")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
class LoadImagesFromDirListAdvanced:
    def __init__(self):
        pass
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "color": ("STRING", {"default": "#FFFFFF"}),
                    "upscale_method": (s.upscale_methods, {"default": "lanczos"}),
                    "target_width": ("INT", {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number"}),
                    "target_height": ("INT", {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_IS_LIST = (True, True)

    FUNCTION = "load_images"

    CATEGORY = "badger"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str,color,upscale_method,target_width,target_height, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= image_load_cap:
                break
            img = Image.open(image_path)
            width = img.size[0]
            height = img.size[1]
            nw = width
            nh = height
            top = 0
            left = 0
            bottom = 0
            right = 0
            if target_width > 0 and target_height > 0 and target_width != width and target_height != height:
                o_ratio = width / height
                ratio = target_width / target_height
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
                image = imgToTensor(img)
                samples = image.movedim(-1,1)
                s = comfy.utils.common_upscale(samples, nw, nh, upscale_method, crop="disabled")
                s = s.movedim(1,-1)
                img = tensorToImg(s)
            if color:
                rgba_color = hex_to_rgba(color)
                new_img = Image.new("RGBA",(nw+left+right,nh+top+bottom), rgba_color)
                new_img.paste(img, (left, top),img.convert("RGBA"))
                img = new_img
            for i in ImageSequence.Iterator(img):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB" if color else "RGBA")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            image_count += 1

        return images, masks

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
    
class IntToStringAdvanced:
 
    def __init__(self):
            pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int": ("INT", {
                    "default": 0,
                    "min": -sys.maxsize - 1,
                    "max": sys.maxsize,
                    "step": 1,
                    "display": "number"
                    }),
                "length": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                    }),
                "prefix":("STRING", {"default": ""}),
                "suffix":("STRING", {"default": ""}),
                },
                
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "int_to_string"

    # OUTPUT_NODE = False

    CATEGORY = "badger"

    def int_to_string(self, int,length,prefix,suffix):
        return (prefix+str(int).zfill(length)+suffix,)


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
                "video_path": ("STRING", {"default": ""}),
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
                "frame_dir": ("STRING", {"default": ""}),
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
    def __init__(self) -> None:
        pass

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
        garbage_collect()
        return (X, Y,)


class SegmentToMaskByPoint:
    def __init__(self) -> None:
        pass

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
        garbage_collect()
        return (mask0, mask1, mask2,)


class CropImageByMask:
    def __init__(self) -> None:
        pass

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
        garbage_collect()
        return (cropped_image, X, Y,)


class ApplyMaskToImage:
    def __init__(self) -> None:
        pass

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
        mask = maskTensorToImgTensor(mask)
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
        garbage_collect()
        return (imgToTensor(result_image),)


class DeleteDir:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("STRING", {"default": ""}),
                "dir_path": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "badger"
    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("result", "e_info")
    FUNCTION = "delete_dir"

    def delete_dir(self, start, dir_path):
        e_info = ""
        status = 0
        abs_dir_path = os.path.abspath(dir_path)
        if not os.path.exists(abs_dir_path):
            e_info = "路径不存在"
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
                status = 1
                e_info = "成功删除"
            except Exception as e:
                status = 0
                e_info = str(e)

        garbage_collect()
        return (status, e_info,)


class FindThickLinesFromCanny:
    def __init__(self) -> None:
        pass

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
        garbage_collect()
        return (result_tensor,)


class TrimTransparentEdges:
    def __init__(self) -> None:
        pass

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
        garbage_collect()
        return (cropped_img,)


class ExpandImageWithColor:
    def __init__(self) -> None:
        pass

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
                "color": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "expand_image_with_color"

    def expand_image_with_color(self, image, top, bottom, left, right, color=None):
        img = tensorToImg(image)
        img = img.convert("RGBA")  # 确保图片是RGBA模式

        # Determine the new size of the image
        new_width = img.width + left + right
        new_height = img.height + top + bottom

        # 如果提供了颜色，并且是十六进制形式，转换为RGBA格式
        if color:
            rgba_color = hex_to_rgba(color)
            new_img = Image.new("RGBA", (new_width, new_height), rgba_color)
        else:
            # Use transparency if no color was provided
            new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Paste the original image onto the new image
        new_img.paste(img, (left, top), img)
        if color:
            new_img = new_img.convert("RGB")
        result = imgToTensor(new_img)
        garbage_collect()
        return (result,)


class GetUUID:
    def __init__(self) -> None:
        pass

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


class GetDirName:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir_path": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_dir_name"

    def get_dir_name(self, dir_path):
        folder_name = os.path.basename(dir_path)
        return (folder_name,)


class GetColorFromBorder:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_width": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "classification_threshold": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_color_from_border"

    def get_color_from_border(self, image, detection_width, classification_threshold):
        pil_img = tensorToImg(image)
        colors = get_colors(pil_img, detection_width)
        color = most_common_fuzzy_color(colors, classification_threshold)
        garbage_collect()
        return (color,)


class IdentifyColorToMask:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#ffffff"}),
                "detection_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "identify_color_to_mask"

    def identify_color_to_mask(self, image, color, detection_threshold):
        pil_img = tensorToImg(image)
        mask_img = find_similar_colors(pil_img, color, detection_threshold)
        mask_tensor = imgToTensor(mask_img)
        mask = img_to_mask(mask_img)
        garbage_collect()
        return (mask_tensor, mask,)


class IdentifyBorderColorToMask:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#ffffff"}),
                "detection_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "identify_border_color_to_mask"

    def identify_border_color_to_mask(self, image, color, detection_threshold):
        pil_img = tensorToImg(image)
        mask_img = detect_outline(pil_img, color, detection_threshold)
        mask_tensor = imgToTensor(mask_img)
        mask = img_to_mask(mask_img)
        garbage_collect()
        return (mask_tensor, mask,)


class GarbageCollect:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("STRING", {"default": "start"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    CATEGORY = "badger"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gc_node"
    OUTPUT_NODE = True

    def gc_node(self, start, seed):
        garbage_collect()
        return (start,)
    

class ToPixel:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "threshold": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "pix": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "tile_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "color_card": ("IMAGE",),
            }
        }

    CATEGORY = "badger"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_pixel"

    def image_to_pixel(self, original_image, threshold, pix, tile_size, color_card=None):
        original_image = tensorToImg(original_image)
        if color_card!=None:
            color_card = tensorToImg(color_card)
        pixelated_image = to_pixel(original_image, threshold, pix, tile_size, color_card)
        pixelated_image = imgToTensor(pixelated_image)
        garbage_collect()
        return (pixelated_image,)

class SimpleBoolean:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "String": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "badger"
    RETURN_TYPES = ("INT",)
    FUNCTION = "simple_boolean"

    def simple_boolean(self,String):
        String = "result = " + String

        # 创建一个空字典用于exec的局部命名空间
        namespace = {}

        # 将namespace字典作为exec的第二个参数，指定局部命名空间
        exec(String, namespace)

        # 从指定的命名空间字典中提取result变量的值
        result = namespace['result']
        if result :
            return (1,)
        else:
            return (0,)

class GETRequset:
    def __init__(self) -> None:
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "params_json": ("STRING",{"default": '{"key1":value1,"key2":"value2"}'}),
                "save_path": ("STRING", {"default": "./output"}),
            },
        }
    CATEGORY = "badger"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_requset"
    OUTPUT_NODE = True

    def get_requset(self,url,params_json,save_path):
        result=""
        json_object = json.loads(params_json)
        response = requests.get(url, params=json_object)
        if response.status_code == 200:
            # 从Content-Disposition头获取文件名
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition:
                filename_start = content_disposition.index('filename=') + 9  # 9是因为'filename='.length()
                filename = content_disposition[filename_start:].strip('"')
            else:
                filename = 'downloaded_file.wav'  # 如果没有指定，默认文件名
            # 确保目录存在
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 指定本地保存路径
            local_filepath = os.path.join(save_path, filename)
            print(local_filepath)
            
            # 保存文件
            with open(local_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return (os.path.abspath(local_filepath), )  # 返回保存的文件路径
        else:
            return (f"请求失败，状态码：{response.status_code}",)
        
class RotateImageWithPadding:
    def __init__(self) -> None:
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
            },
        }
    CATEGORY = "badger"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_and_pad_image"
    OUTPUT_NODE = True

    def rotate_and_pad_image(self,original_image):
        img_PIL = tensorToImg(original_image)
        result = rotate_image_with_padding(img_PIL)
        img_tensor = imgToTensor(result)
        garbage_collect()
            
        return (img_tensor,)
        

NODE_CLASS_MAPPINGS = {
    "ImageOverlap-badger": ImageOverlap,
    "FloatToInt-badger": FloatToInt,
    "IntToString-badger": IntToString,
    "LoadImageAdvanced-badger": LoadImageAdvanced,
    "LoadImagesFromDirListAdvanced-badger":LoadImagesFromDirListAdvanced,
    "IntToStringAdvanced-badger":IntToStringAdvanced,
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
    "GetUUID-badger": GetUUID,
    "GetDirName-badger": GetDirName,
    "GetColorFromBorder-badger": GetColorFromBorder,
    "IdentifyColorToMask-badger":IdentifyColorToMask,
    "IdentifyBorderColorToMask-badger":IdentifyBorderColorToMask,
    "GarbageCollect-badger": GarbageCollect,
    "ToPixel-badger": ToPixel,
    "SimpleBoolean-badger": SimpleBoolean,
    "GETRequset-badger": GETRequset,
    "RotateImageWithPadding":RotateImageWithPadding

}

NODE_DISPLAY_NAME_MAPPINGS = {
}
