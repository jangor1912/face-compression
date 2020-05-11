import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image

from face_detection.face_landmarks import FaceLandmarksPredictorFAN


class FaceMetric(object):
    def __init__(self, predictor, output_size=128):
        self.predictor = predictor
        self.size = output_size
        self.white = (255, 255, 255)
        self.gray = (128, 128, 128)
        self.middle = (192, 192, 192)

    @staticmethod
    def get_loss(y_true, y_pred):
        produced_image = y_pred
        original_image = y_true[0]
        mask = y_true[1]
        mask += 1.0  # This means that mask values belong to <0, 2>
        mse_tensor = tf.square(tf.subtract(original_image, produced_image))
        return tf.reduce_mean(tf.multiply(mse_tensor, mask))

    @classmethod
    def get_loss_from_batch(cls, y_true_batch, y_pred_batch):
        result = list()
        for i in range(int(y_pred_batch.shape[0])):
            result.append(cls.get_loss(y_true_batch[i], y_pred_batch[i]))
        return tf.reduce_mean(result)

    def generate_mask(self, face_prediction, img_height=None, img_width=None):
        img_height = img_height or self.size
        img_width = img_width or self.size
        face_width, face_height = self.predictor.get_face_dimensions(face_prediction)

        # draw background
        mask = Image.new('RGB', (img_width, img_height), color=self.gray)
        draw = ImageDraw.Draw(mask)
        if face_prediction is None:
            mask = np.array(mask)
            return mask

        # draw face
        draw.polygon(face_prediction[self.predictor.pred_types["face"].slice],
                     outline=self.middle, fill=self.middle)
        # draw circle to include forehead
        first = face_prediction[self.predictor.pred_types["face"].slice.start]
        last = face_prediction[self.predictor.pred_types["face"].slice.stop - 1]
        radius = np.math.fabs(last[0] - first[0]) / 2
        upper_left = (first[0], first[1] - radius)
        down_right = (last[0], last[1] + radius)
        try:
            draw.ellipse([upper_left, down_right],
                         outline=self.middle, fill=self.middle)
        except Exception as e:
            print(f"Exception during ellipse drawing occurred. Exception = {str(e)}")

        # draw eyebrows
        draw.polygon(face_prediction[self.predictor.pred_types["eyebrow1"].slice],
                     fill=self.white, outline=self.white)
        draw.polygon(face_prediction[self.predictor.pred_types["eyebrow2"].slice],
                     fill=self.white, outline=self.white)

        # draw eyes
        draw.polygon(face_prediction[self.predictor.pred_types["eye1"].slice],
                     fill=self.white, outline=self.white)
        draw.polygon(face_prediction[self.predictor.pred_types["eye2"].slice],
                     fill=self.white, outline=self.white)

        # draw nose
        draw.line(face_prediction[self.predictor.pred_types["nostril"].slice],
                  fill=self.white, width=math.ceil(face_height / 50))
        draw.line(face_prediction[self.predictor.pred_types["nose"].slice],
                  fill=self.white, width=math.ceil(face_height / 50))

        # draw lips
        draw.polygon(face_prediction[self.predictor.pred_types["lips"].slice],
                     outline=self.white, fill=self.white)
        draw.line(face_prediction[self.predictor.pred_types["teeth"].slice],
                  fill=self.white, width=math.ceil(face_height / 100))

        mask = np.array(mask)
        return mask


def main():
    image_path = "/home/jan/PycharmProjects/face-compression/data/images/my_face_front.png"
    predictor = FaceLandmarksPredictorFAN(_type='2d')
    image = cv2.imread(image_path)
    img_height, img_width, _ = np.array(image).shape
    metric = FaceMetric(predictor)
    prediction = predictor.detect_one_face_landmark(image)
    mask = metric.generate_mask(prediction, img_height=img_height, img_width=img_width)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot()
    ax.imshow(np.array(mask))
    plt.show()


if __name__ == "__main__":
    main()
