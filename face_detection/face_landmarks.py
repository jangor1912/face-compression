import argparse

import cv2
import dlib
import face_alignment
import imutils
import numpy as np

from config.config import Config
from face_detection.misc.image_helpers import ImageHelpers

CONF = Config().CONF


class FaceLandmarksPredictor(object):
    def __init__(self, model_path=None):
        self.predictor_model_path = CONF["path"]["face_landmarks_model"] if not model_path else model_path

    @classmethod
    def shape_to_np(cls, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def place_landmarks(self, image, image_path=True):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(self.predictor_model_path))

        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image) if image_path else image
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = self.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            # (x, y, w, h) = rect_to_bb(rect)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        return image


class FaceLandmarksPredictorFAN(object):
    def __init__(self, device='cpu'):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    def place_landmarks(self, image, predictions):
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        if predictions:
            for prediction in predictions:
                for x, y in prediction:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        return image

    def detect_landmarks(self, image):
        img_height, img_width, _ = np.array(image).shape
        final_predictions = []
        predictions = self.fa.get_landmarks(image)
        if predictions:
            for prediction in predictions:
                current_prediction = []
                for x, y in prediction:
                    if 0 <= x <= img_height and 0 <= y <= img_width:
                        current_prediction.append((x, y))
                if current_prediction:
                    final_predictions.append(current_prediction)
        return final_predictions

    def detect_face(self, image, predictions):
        min_x = 6000
        min_y = 6000
        max_x = -1
        max_y = -1
        for prediction in predictions:
            for x, y in prediction:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # expand image
        top, left, bottom, right = min_y, min_x, max_y, max_x
        face_location = (top, left, bottom, right)

        # make crop square
        face_location = ImageHelpers.get_square(image, face_location)
        face_location = ImageHelpers.expand_image(image, face_location, expansion_factor=0.1)
        return face_location


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())
    image = FaceLandmarksPredictor().place_landmarks(args["image"], args["shape-predictor"])
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_path = "/home/jan/PycharmProjects/face-compression/data/images/my_face.png"
    predictor = FaceLandmarksPredictorFAN()
    # image = place_landmarks(image_path)
    image = cv2.imread(image_path)
    predictions = predictor.detect_landmarks(image)
    image = predictor.place_landmarks(image, predictions)
    face_location = predictor.detect_face(image, predictions)
    image = ImageHelpers.crop_image(image, face_location)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)
