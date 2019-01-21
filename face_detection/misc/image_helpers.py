import math
import os

import numpy as np


class ImageHelpers(object):
    @staticmethod
    def touch(path):
        with open(path, 'a'):
            os.utime(path, None)

    @classmethod
    def get_diagonal(cls, position_tuple):
        top, right, bottom, left = position_tuple
        height = bottom - top
        width = right - left
        diagonal_len = math.sqrt(math.pow(height, 2) + math.pow(width, 2))
        return diagonal_len

    @classmethod
    def expand_image(cls, image, face_location, expansion_factor=0.2):
        img_height, img_width, _ = np.array(image).shape
        top, right, bottom, left = face_location
        diagonal_len = cls.get_diagonal(face_location)
        expansion = int(math.floor(expansion_factor * diagonal_len))
        if top-expansion > 0 and left-expansion > 0:
            top -= expansion
            left -= expansion
        if bottom+expansion < img_height and right+expansion < img_width:
            bottom += expansion
            right += expansion
        return top, right, bottom, left
