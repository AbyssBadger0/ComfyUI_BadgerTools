
from PIL import Image
import numpy as np
import torch
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

    def tensorToImg(self, imageTensor):
        imaget = imageTensor[0]
        i = 255. * imaget.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def imgToTensor(self,img):
        image = np.array(img).astype(np.float32) / 255.0
        imaget = torch.from_numpy(image)[None,]
        return imaget

    def overlap(self, base_image, additional_image, x, y):
        b_image = self.tensorToImg(base_image)
        a_image = self.tensorToImg(additional_image)

        b_image.paste(a_image, (x, y))
        o_image = self.imgToTensor(b_image)
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


NODE_CLASS_MAPPINGS = {
    "ImageOverlap-badger": ImageOverlap,
    "FloatToInt-badger": FloatToInt,
    "IntToString-badger": IntToString,
    "FloatToString-badger": FloatToString

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOverlap": "Example test"
}
