import math
from pathlib import Path

import cv2
import numpy as np

from config.config import Config
from face_detection.face_detector import FaceDetector
from face_detection.face_landmarks import FaceLandmarksPredictorFAN
from face_detection.misc.image_helpers import ImageHelpers
from video.deconstructor.deconstructor import Deconstructor

CONF = Config().CONF


def generate_landmarked_face_video(input_path, output_path, output_size=512):
    if not Path(input_path).exists():
        raise RuntimeError("Input path = {} does not exist!".format(input_path))

    frame_generator = Deconstructor().video_to_images(input_path)
    face_detector = FaceDetector(model="hog")
    predictor = FaceLandmarksPredictorFAN()
    size = (output_size, output_size)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 30, size)

    # parse first frame
    frame, _ = frame_generator.__next__()
    img_height, img_width, _ = np.array(frame).shape

    # resizing for speedup
    # frame = cv2.resize(frame, (int(img_width * 0.3), int(img_height * 0.3)), interpolation=cv2.INTER_AREA)
    # cv2.imshow("output", frame)
    # cv2.waitKey(30)

    face_locations = face_detector.detect_faces(frame)
    face_location = None
    if face_locations:
        face_location = ImageHelpers.get_square(frame, face_locations[0])

    last_goal = face_location or [int(img_height / 3), int(img_height / 3), int(img_height * 2 / 3),
                                  int(img_height * 2 / 3)]
    last_location = last_goal
    last_movement_speed = (0, 0)  # in pixels per frame
    last_resize_speed = 0  # in pixels per frame

    resize_smoothness = 4.0
    movement_smoothness = 4.0
    for frame, frame_no in frame_generator:
        print("Processing frame = {}".format(frame_no))
        # resizing for speedup
        # img_height, img_width, _ = np.array(frame).shape
        # frame = cv2.resize(frame, (int(img_width * 0.3), int(img_height * 0.3)), interpolation=cv2.INTER_AREA)

        predictions = predictor.detect_landmarks(frame)
        image = predictor.place_landmarks(frame, predictions)
        face_locations = face_detector.detect_faces(frame)
        if face_locations:
            face_location = ImageHelpers.get_square(image, face_locations[0])
        else:
            face_location = last_goal
        last_goal = face_location

        optimal_location_center = ImageHelpers.get_location_center(face_location)
        last_location_center = ImageHelpers.get_location_center(last_location)

        optimal_movement_speed = [
            math.ceil((optimal_location_center[0] - last_location_center[0]) / movement_smoothness),
            math.ceil((optimal_location_center[1] - last_location_center[1]) / movement_smoothness)]

        current_movement_speed = [int((last_movement_speed[0] + optimal_movement_speed[0]) / 2),
                                  int((last_movement_speed[1] + optimal_movement_speed[1]) / 2)]
        last_movement_speed = current_movement_speed
        # current_movement_speed = (max(min_pixels_per_frame, item) for i, item in enumerate(current_movement_speed))

        optimal_diagonal = ImageHelpers.get_diagonal(face_location)
        last_diagonal = ImageHelpers.get_diagonal(last_location)

        optimal_resize_speed = math.ceil((optimal_diagonal - last_diagonal) / 5.0)
        current_resize_speed = int((last_resize_speed + optimal_resize_speed) / resize_smoothness)
        last_resize_speed = current_resize_speed

        current_location = (int(last_location[0] + current_movement_speed[1] - current_resize_speed / 2),
                            int(last_location[1] + current_movement_speed[0] - current_resize_speed / 2),
                            int(last_location[2] + current_movement_speed[1] + current_resize_speed / 2),
                            int(last_location[3] + current_movement_speed[0] + current_resize_speed / 2))
        last_location = current_location
        print("\tCurrent location = {}".format(current_location))
        print("\tFace location = {}".format(face_location))
        print("\tCurrent movement speed = {}".format(current_movement_speed))
        print("\tOptimal movement speed = {}".format(optimal_movement_speed))
        print("\tCurrent resize speed = {}".format(current_resize_speed))
        print("\tOptimal resize speed = {}".format(optimal_resize_speed))

        # current_location = ImageHelpers.get_square(image, current_location)
        image = ImageHelpers.crop_image(frame, current_location)
        image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        cv2.imshow("output", image)
        cv2.waitKey(30)

        video_writer.write(image)
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "/home/jan/PycharmProjects/face-compression/data/videos/head_movement.mp4"
    video_output_path = video_path.rstrip(".mp4") + "-landmarks-fan-predictor-face-only-following.avi"
    generate_landmarked_face_video(video_path, video_output_path)
    # resize_width = 512
    # if Path(video_path).exists():
    #     print("Output path = {} exists!".format(video_path))
    # else:
    #     print("Output path = {} does not exist!".format(video_path))
    #     sys.exit()
    #
    # frame_generator = Deconstructor().video_to_images(video_path)
    # predictor = FaceLandmarksPredictorFAN()
    # first_frame, _ = frame_generator.__next__()
    # image = predictor.place_landmarks(first_frame)
    # height, width, layers = image.shape
    # scale = float(resize_width) / float(width)
    # size = (resize_width, int(height * scale))
    # video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 30, size)
    # for frame, frame_no in frame_generator:
    #     image = predictor.place_landmarks(frame, resize_width=resize_width)
    #     # cv2.imshow("output", image)
    #     # cv2.waitKey(30)
    #
    #     video_writer.write(image)
    # # cv2.destroyAllWindows()
    # video_writer.release()
