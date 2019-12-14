import dlib

from config.config import Config
from face_detection.misc.image_helpers import ImageHelpers

CONF = Config().CONF


class FaceDetector(object):
    def __init__(self, model="hog"):
        if model == "hog":
            self.face_detector = dlib.get_frontal_face_detector()
        else:
            self.face_detector = dlib.cnn_face_detection_model_v1(CONF["path"]["cnn_face_model"])

    def detect_faces(self, image):
        rectangles = self.face_detector(image, 1)
        face_locations = []
        for rectangle in rectangles:
            face_locations.append((rectangle.top(), rectangle.right(),
                                   rectangle.bottom(), rectangle.left()))

        locations = []
        for face_location in face_locations:
            location = ImageHelpers.expand_image(image, face_location)
            locations.append(location)
        return locations


if __name__ == "__main__":
    import cv2
    image_path = "D:\\Pictures\\4K Stogram\\face_tag\\face.jpg"
    image = cv2.imread(image_path)
    face_detector = FaceDetector(model="cnn")
    for face in face_detector.detect_faces(image):
        print(str(face))
        top, right, bottom, left = face
        cv2.rectangle(image, (right, top), (left, bottom), (255, 255, 255), 3)
    cv2.imshow('Video', image)
    cv2.waitKey(0)

