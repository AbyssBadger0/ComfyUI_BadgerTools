import math

from PIL import Image
import numpy as np
import torch


def tensorToImg(imageTensor):
    imaget = imageTensor[0]
    i = 255. * imaget.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


def imgToTensor(img):
    image = np.array(img).astype(np.float32) / 255.0
    imaget = torch.from_numpy(image)[None,]
    return imaget

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
                "width": ("INT",  {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "height": ("INT",  {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "target_width": ("INT",  {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4096.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"}),
                "target_height": ("INT",  {
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
        o_ratio = width/height
        ratio = target_width/target_height
        top = 0
        left = 0
        bottom = 0
        right = 0
        nw = 0
        nh = 0
        # 原图比期望尺寸更扁，对齐宽，计算高，补上下
        if(o_ratio>=ratio):
            upratio = target_width/width
            nw = target_width
            nh = round(height*upratio)
            hdiff = target_height - nh
            top = math.floor(hdiff/2)
            bottom = math.ceil(hdiff/2)
        else:
            upratio = target_height/height
            nw = round(width*upratio)
            nh = target_height
            wdiff = target_width - nw
            left = math.floor(wdiff/2)
            right = math.ceil(wdiff/2)

        return (nw, nh, top, left, bottom, right,)






NODE_CLASS_MAPPINGS = {
    "ImageOverlap-badger": ImageOverlap,
    "FloatToInt-badger": FloatToInt,
    "IntToString-badger": IntToString,
    "FloatToString-badger": FloatToString,
    "ImageNormalization-badger": ImageNormalization
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOverlap": "Example test"
}
