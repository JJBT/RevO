import PIL.Image
import numpy as np
from PIL.Image import Image
from demo.convert_utils import base64_to_image, array_to_base64


class ImageInput:
    def __init__(self, shape=None, name=None):
        self.shape = shape
        self.name = name

    def get_context(self):
        return {'image_mode': 'RGB', 'shape': self.shape, 'name': self.name}

    def preprocess(self, input):
        input = base64_to_image(input)
        input = input.convert('RGB')
        input = np.array(input)
        return input


class ImageOutput:
    def __init__(self, name=None):
        self.name = name

    @classmethod
    def get_shortcut(cls):
        return {
            'image': {}
        }

    def get_context(self):
        return {
            'name': self.name,
        }

    def postprocess(self, img):
        # img = PIL.Image.fromarray(np.array(img, np.uint8))
        return array_to_base64(img)
