import math
from pathlib import Path

import cv2

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

    last_goal = (0, 0, output_size, output_size)
    last_location = (0, 0, output_size, output_size)
    last_movement_speed = (0, 0)  # in pixels per frame
    last_resize_speed = 0  # in pixels per frame
    for frame, frame_no in frame_generator:
        print("Processing frame = {}".format(frame_no))
        face_locations = face_detector.detect_faces(frame)
        face_location = face_locations[0] if face_locations else last_goal
        last_goal = face_location

        optimal_location_center = ImageHelpers.get_location_center(face_location)
        last_location_center = ImageHelpers.get_location_center(last_location)

        optimal_movement_speed = (math.ceil((optimal_location_center[0] - last_location_center[0])/5.0),
                                  math.ceil((optimal_location_center[1] - last_location_center[1])/5.0))

        current_movement_speed = (int((last_movement_speed[0] + optimal_movement_speed[0])/2),
                                  int((last_movement_speed[1] + optimal_movement_speed[1])/2))

        current_diagonal = ImageHelpers.get_diagonal(face_location)
        last_diagonal = ImageHelpers.get_diagonal(last_location)

        optimal_resize_speed = math.ceil((current_diagonal - last_diagonal)/5.0)
        current_resize_speed = int((last_resize_speed + optimal_resize_speed)/2)

        current_location = (int(last_location[0] + current_movement_speed[0] + current_resize_speed / 2),
                            int(last_location[1] + current_movement_speed[1] + current_resize_speed / 2),
                            int(last_location[2] + current_movement_speed[0] + current_resize_speed / 2),
                            int(last_location[3] + current_movement_speed[1] + current_resize_speed / 2))
        print("Current location = {}".format(current_location))

        image = ImageHelpers.crop_image(frame, current_location)
        predictions = predictor.detect_landmarks(image)
        image = predictor.place_landmarks(image, predictions)
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
