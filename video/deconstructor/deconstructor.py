import os

import cv2

from config.config import Config
from face_detection.face_detector import FaceDetector
from face_detection.misc.image_helpers import ImageHelpers

CONF = Config().CONF


class Deconstructor(object):
    def __init__(self):
        self.face_detector = FaceDetector()

    def after_download(self, stream, file_handle):
        video_path = file_handle.name
        self.find_faces_in_video(video_path)
        os.remove(video_path)

    def video_to_images(self, video_path, start_frame=0):
        vidcap = cv2.VideoCapture(video_path)
        try:
            if start_frame:
                success = vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                if not success:
                    raise RuntimeError(f"Cannot set frame of video {video_path} to {start_frame}")
        except Exception as e:
            message = f"Error during skipping frames. Error = {str(e)}"
            raise RuntimeError(str(message))
        success, image = vidcap.read()
        count = 0
        while success:
            yield image, count
            success, image = vidcap.read()
            count += 1
        vidcap.release()

    def find_faces_in_video(self, video_path, skip=60, threshold=5):
        directory_path = os.path.join(CONF["directory"]["faces"], os.path.basename(video_path))
        try:
            os.mkdir(directory_path)
        except FileExistsError:
            pass

        currently_skipped = 0
        probably_face = False
        face_locations = []
        current_threshold = 0
        uncertain_images = []
        for img_tuple in self.video_to_images(video_path):
            if currently_skipped == skip:
                currently_skipped = 0
                image_array, count = img_tuple
                face_locations = self.face_detector.detect_faces(image_array)
                if face_locations:
                    probably_face = True
                    current_threshold = threshold
                    uncertain_images = []

            if probably_face or current_threshold > 0:
                image_array, count = img_tuple
                img_path = os.path.join(directory_path, "frame{}.jpg".format(count))
                print("Current image path = {}".format(img_path))

                if not face_locations:
                    face_locations = self.face_detector.detect_faces(image_array)

                if face_locations:
                    # Pick first face for simplicity
                    face_location = face_locations[0]
                    probably_face = True
                    current_threshold = threshold
                    face_locations = []
                    uncertain_images = []
                else:
                    probably_face = False
                    current_threshold -= 1
                    uncertain_images.append(img_path)

                # Face location must exist at this point
                face_image = ImageHelpers.crop_image(image_array, face_location)

                cv2.imwrite(img_path, face_image)
            else:
                # Image is not a face and threshold was exhausted
                current_threshold = 0
                probably_face = False
                face_locations = []
                face_location = None

                # Cleaning up all images that were uncertain
                for image_path in uncertain_images:
                    if os.path.isfile(image_path):
                        os.remove(image_path)

            currently_skipped += 1


if __name__ == "__main__":
    d = Deconstructor()
    d.find_faces_in_video('/home/osboxes/PycharmProjects/face-compression/Follow me around high school vlog 2019.mp4')


