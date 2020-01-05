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
        top, left, bottom, right = position_tuple
        height = bottom - top
        width = right - left
        diagonal_len = math.sqrt(math.pow(height, 2) + math.pow(width, 2))
        return diagonal_len

    @classmethod
    def expand_image(cls, image, face_location, expansion_factor=0.2):
        img_height, img_width, _ = np.array(image).shape
        top, left, bottom, right = face_location
        diagonal_len = cls.get_diagonal(face_location)
        expansion = int(math.floor(expansion_factor * diagonal_len))
        if top - expansion > 0 and left - expansion > 0:
            top -= expansion
            left -= expansion
        else:
            best_expansion = min(top, left)
            top -= best_expansion
            left -= best_expansion

        if bottom + expansion < img_height and right + expansion < img_width:
            bottom += expansion
            right += expansion
        else:
            best_expansion = min(img_height - bottom, img_width - right)
            bottom += best_expansion
            right += best_expansion
        return int(top), int(left), int(bottom), int(right)

    @classmethod
    def crop_image(cls, image_array, locations):
        locations = [int(item) for item in locations]
        top, left, bottom, right = locations
        cropped_image = image_array[top:bottom, left:right]
        return cropped_image

    @classmethod
    def get_square(cls, image, locations):
        top, left, bottom, right = locations
        height = bottom - top
        width = right - left
        img_height, img_width, _ = np.array(image).shape

        difference = abs(height - width)
        expansion = int(difference / 2)
        if height > width:
            if left - expansion >= 0:
                left -= expansion
            if right + expansion <= img_width:
                right += expansion

            width = right - left
            if width != height:
                difference = abs(height - width)
                bottom -= difference
        else:
            if top - expansion >= 0:
                top -= expansion
            if bottom + expansion <= img_height:
                bottom += expansion

            height = bottom - top
            if width != height:
                difference = abs(height - width)
                right -= difference
        locations = top, left, bottom, right
        return locations

    @classmethod
    def get_location_center(cls, location):
        top, left, bottom, right = location
        x_center = int((right + left) / 2)
        y_center = int((top + bottom) / 2)
        return x_center, y_center

    @classmethod
    def get_image_center(cls, image_array):
        img_height, img_width, _ = np.array(image_array).shape
        y_center = int(img_height / 2)
        x_center = int(img_width / 2)
        return x_center, y_center
