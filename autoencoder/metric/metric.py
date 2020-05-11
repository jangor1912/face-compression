import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image, ImageOps, ImageFilter

# from face_detection.face_landmarks import FaceLandmarksPredictorFAN
from dataset.batch_generator import BatchSequence


class FaceMetric(object):
    def __init__(self, predictor, output_size=128):
        self.predictor = predictor
        self.size = output_size
        self.white = (16, 16, 16, 128)

    @staticmethod
    def get_loss(y_true, y_pred):
        produced_image = y_pred
        original_image = y_true[0]
        mask = y_true[1]
        mask += 2.0  # This means that mask values belong to <1, 3>
        mse_tensor = tf.square(tf.subtract(original_image, produced_image))
        result = tf.reduce_mean(tf.multiply(mse_tensor, mask))
        return result * 100.0

    @classmethod
    def get_loss_from_batch(cls, y_true_batch, y_pred_batch):
        result = list()
        for i in range(int(y_pred_batch.shape[0])):
            result.append(cls.get_loss(y_true_batch[i], y_pred_batch[i]))
        return tf.reduce_mean(result)

    def generate_mask(self, face_prediction, img_height=None, img_width=None):
        result = None
        img_height = img_height or self.size
        img_width = img_width or self.size
        images = {}

        # draw background
        if face_prediction is None:
            mask = Image.new('RGB', (img_width, img_height), color=(128, 128, 128))
            mask = np.array(mask)
            return mask

        # draw face
        face = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(face)
        draw.polygon(face_prediction[self.predictor.pred_types["face"].slice],
                     outline=self.white, fill=self.white)
        # draw circle to include forehead
        first = face_prediction[self.predictor.pred_types["face"].slice.start]
        last = face_prediction[self.predictor.pred_types["face"].slice.stop - 1]
        radius = np.math.fabs(last[0] - first[0]) / 2
        upper_left = (first[0], first[1] - radius)
        down_right = (last[0], last[1] + radius)
        draw.ellipse([upper_left, down_right],
                     outline=self.white, fill=self.white)
        images["face"] = face

        # draw eyebrows
        eyebrows = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(eyebrows)
        draw.line(face_prediction[self.predictor.pred_types["eyebrow1"].slice],
                  fill=self.white, width=int(img_height / 50))
        draw.line(face_prediction[self.predictor.pred_types["eyebrow2"].slice],
                  fill=self.white, width=int(img_height / 50))
        images["eyebrows"] = eyebrows

        # draw eyes
        eye1 = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(eye1)
        points = face_prediction[self.predictor.pred_types["eye1"].slice]
        draw.polygon(points, outline=self.white, fill=self.white)
        draw.line(points, fill=self.white, width=int(img_height / 50))
        images["eye1"] = eye1

        eye2 = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(eye2)
        points = face_prediction[self.predictor.pred_types["eye2"].slice]
        draw.polygon(points, outline=self.white, fill=self.white)
        draw.line(points, fill=self.white, width=int(img_height / 50))
        images["eye2"] = eye2

        # draw nose
        nose = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(nose)
        draw.line(face_prediction[self.predictor.pred_types["nostril"].slice],
                  fill=self.white, width=int(img_height / 50))
        draw.line(face_prediction[self.predictor.pred_types["nose"].slice],
                  fill=self.white, width=int(img_height / 50))
        images["nose"] = nose

        # draw lips
        lips = Image.new('RGBA', (img_width, img_height))
        draw = ImageDraw.Draw(lips)
        points = face_prediction[self.predictor.pred_types["lips"].slice]
        draw.polygon(points, outline=self.white, fill=self.white)
        draw.line(points, fill=self.white, width=int(img_height / 50))
        draw.line(face_prediction[self.predictor.pred_types["teeth"].slice],
                  fill=self.white, width=int(img_height / 100))
        images["lips"] = lips

        mask = Image.new('RGBA', (img_width, img_height), color=self.white)
        for image in images.values():
            mask.alpha_composite(image)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=int(img_height / 100)))

        background = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        background.paste(mask, mask=mask.split()[3])  # 3 is the alpha channel
        mask = ImageOps.invert(background)
        mask = np.array(mask)
        return mask
        # fig = plt.figure(figsize=plt.figaspect(.5))
        # ax = fig.add_subplot()
        # ax.imshow(np.array(mask))
        # plt.show()


def main():
    pass
    # image_path = "/home/jan/PycharmProjects/face-compression/data/images/my_face_front.png"
    # predictor = FaceLandmarksPredictorFAN(_type='2d')
    # image = cv2.imread(image_path)
    # img_height, img_width, _ = np.array(image).shape
    # metric = FaceMetric(predictor)
    # prediction = predictor.detect_one_face_landmark(image)
    # metric.generate_mask(prediction, img_height=img_height, img_width=img_width)


if __name__ == "__main__":
    main()
