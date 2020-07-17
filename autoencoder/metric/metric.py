import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image

from dataset.batch_generator import BatchSequence


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
        mask += 2.0  # This means that mask values belong to <1, 3>
        mse_tensor = tf.square(tf.subtract(original_image, produced_image))
        result = tf.reduce_mean(tf.multiply(mse_tensor, mask))
        return result

    @classmethod
    def get_loss_from_batch(cls, y_true_batch, y_pred_batch):
        result = list()
        for i in range(int(y_pred_batch.shape[0])):
            result.append(cls.get_loss(y_true_batch[i], y_pred_batch[i]))
        return tf.reduce_mean(result)

    def generate_mask(self, face_prediction, img_height=None, img_width=None):
        img_height = img_height or self.size
        img_width = img_width or self.size

        # draw background
        mask = Image.new('RGB', (img_width, img_height), color=self.gray)
        draw = ImageDraw.Draw(mask)
        if face_prediction is None:
            mask = np.array(mask)
            return mask

        face_width, face_height = self.predictor.get_face_dimensions(face_prediction)
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


def test_metric(correct_img: Path,
                mask_img: Path,
                broken_face_img: Path,
                broken_background_img: Path):
    correct_img = Image.open(correct_img)
    mask_img = Image.open(mask_img)
    broken_face_img = Image.open(broken_face_img)
    broken_background_img = Image.open(broken_background_img)

    correct_img = BatchSequence.rgb_image_to_np_array(correct_img)
    mask_img = BatchSequence.rgb_image_to_np_array(mask_img)
    broken_face_img = BatchSequence.rgb_image_to_np_array(broken_face_img)
    broken_background_img = BatchSequence.rgb_image_to_np_array(broken_background_img)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        y_true = np.array([correct_img, mask_img])
        y_pred = np.array([broken_face_img, broken_face_img])
        broken_face_value = FaceMetric.get_loss(y_true, y_pred)

        print("Broken face value = ")
        print(broken_face_value.eval())

    sess = tf.compat.v1.Session()
    with sess.as_default():
        y_true = np.array([correct_img, mask_img])
        y_pred = np.array([broken_background_img, broken_background_img])
        broken_background_value = FaceMetric.get_loss(y_true, y_pred)

        print("Broken background value = ")
        print(broken_background_value.eval())


# def main():
#     image_path = "/home/jan/PycharmProjects/face-compression/data/images/my_face_front.png"
#     predictor = FaceLandmarksPredictorFAN(_type='2d')
#     image = cv2.imread(image_path)
#     img_height, img_width, _ = np.array(image).shape
#     metric = FaceMetric(predictor)
#     prediction = predictor.detect_one_face_landmark(image)
#     mask = metric.generate_mask(prediction, img_height=img_height, img_width=img_width)
#     fig = plt.figure(figsize=plt.figaspect(.5))
#     ax = fig.add_subplot()
#     ax.imshow(np.array(mask))
#     plt.show()


if __name__ == "__main__":
    correct_image = Path("C:/PyCharm Projects/face-compression/data/images/original_img.png")
    mask_image = Path("C:/PyCharm Projects/face-compression/data/images/mask_img.png")
    broken_face_image = Path("C:/PyCharm Projects/face-compression/data/images/broken_face.png")
    broken_background_image = Path("C:/PyCharm Projects/face-compression/data/images/broken_background.png")
    test_metric(correct_image,
                mask_image,
                broken_face_image,
                broken_background_image)
