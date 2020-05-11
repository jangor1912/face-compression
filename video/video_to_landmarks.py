import math
from pathlib import Path

import cv2
import numpy as np

from autoencoder.metric.metric import FaceMetric
from config.config import Config
from face_detection.face_landmarks import FaceLandmarksPredictorFAN
from face_detection.misc.image_helpers import ImageHelpers
from video.deconstructor.deconstructor import Deconstructor

CONF = Config().CONF


def generate_landmarked_face_video(input_path, output_path, output_size=512, strict=False, length=10):
    if not Path(input_path).exists():
        raise RuntimeError("Input path = {} does not exist!".format(input_path))

    frame_generator = Deconstructor.video_to_images(input_path)
    predictor = FaceLandmarksPredictorFAN(device="cpu")
    face_metric = FaceMetric(predictor)
    size = (output_size, output_size)
    video_writer = cv2.VideoWriter(output_path + ".avi", cv2.VideoWriter_fourcc(*"XVID"), 30, size)
    video_writer1 = cv2.VideoWriter(output_path + "-real.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, size)

    # parse first frame
    frame, _ = next(frame_generator)
    img_height, img_width, _ = np.array(frame).shape

    prediction = predictor.detect_one_face_landmark(frame)
    face_location = None
    if prediction:
        face_location = predictor.get_face_square(frame, prediction)
        face_location = ImageHelpers.get_square(frame, face_location)
        face_location = ImageHelpers.expand_image(frame, face_location, expansion_factor=0.3)

    last_goal = face_location or [int(img_height / 3), int(img_height / 3), int(img_height * 2 / 3),
                                  int(img_height * 2 / 3)]
    last_location = last_goal
    last_movement_speed = [0, 0]  # in pixels per frame
    last_resize_speed = 0  # in pixels per frame

    resize_smoothness = 4.0
    movement_smoothness = 4.0

    face_continuity = []
    face_present = False
    length_in_frames = int(30 * length)
    try:
        for frame, frame_no in frame_generator:
            print("Processing frame = {}".format(frame_no))
            # resizing for speedup
            img_height, img_width, _ = np.array(frame).shape

            prediction = predictor.detect_one_face_landmark(frame)

            # fill face metadata -> sections that include face
            if prediction is not None:
                if not face_continuity:
                    face_continuity.append([frame_no, frame_no])
                    print(f"Begin face scene at {frame_no}")
                elif not face_present:
                    face_continuity.append([frame_no, frame_no])
                    print(f"New face scene at {frame_no}")
                else:
                    face_continuity[-1][-1] = frame_no
                face_present = True
            else:
                print(f"End of face scene at {frame_no}. "
                      f"It lasted from {face_continuity[-1][0]} to {face_continuity[-1][1]}")
                face_present = False
            # predictions = [prediction] if prediction else None
            # image1 = predictor.place_landmarks(frame, predictions)
            image1 = frame
            image = face_metric.generate_mask(prediction, img_height=img_height, img_width=img_width)
            if prediction:
                face_location = predictor.get_face_square(frame, prediction)
                face_location = ImageHelpers.get_square(image, face_location)
                face_location = ImageHelpers.expand_image(image, face_location, expansion_factor=0.3)
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
            # print("\tCurrent location = {}".format(current_location))
            # print("\tFace location = {}".format(face_location))
            # print("\tCurrent movement speed = {}".format(current_movement_speed))
            # print("\tOptimal movement speed = {}".format(optimal_movement_speed))
            # print("\tCurrent resize speed = {}".format(current_resize_speed))
            # print("\tOptimal resize speed = {}".format(optimal_resize_speed))

            # current_location = ImageHelpers.get_square(image, current_location)
            image = ImageHelpers.crop_image(image, current_location)
            image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)

            image1 = ImageHelpers.crop_image(image1, current_location)
            image1 = cv2.resize(image1, (output_size, output_size), interpolation=cv2.INTER_AREA)
            # cv2.imshow("output", image)
            # cv2.waitKey(30)

            video_writer.write(image)
            video_writer1.write(image1)
            if frame_no > length_in_frames:
                if not strict:
                    return face_continuity
                if sum([last - first for first, last in face_continuity]) > length_in_frames:
                    return face_continuity
    except Exception as e:
        print(str(e))
    finally:
        video_writer.release()
        video_writer1.release()


def cut_face_lacking_scenes(video_path, output_path, metadata, size=(512, 512)):
    video_writer = cv2.VideoWriter(output_path + ".avi", cv2.VideoWriter_fourcc(*"XVID"), 30, size)
    for start_frame, end_frame in metadata:
        image_generator = Deconstructor.video_to_images(video_path, start_frame=start_frame)
        scene_length = end_frame - start_frame
        for frame, frame_no in image_generator:
            if frame_no > scene_length:
                break
            video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    video_path = "/media/jan/Elements SE/Magisterka/youtube_dataset/schafter - bigos (feat Taco Hemingway).mp4"
    video_output_path = video_path.rstrip(".mp4") + "-metric-mask"
    metadata = generate_landmarked_face_video(video_path, video_output_path, length=1, strict=True)
    split_video_output_path = video_path.rstrip(".mp4") + "-split"
    split_mask_video_output_path = video_path.rstrip(".mp4") + "-split-mask"
    cut_face_lacking_scenes(video_output_path + ".avi", split_mask_video_output_path, metadata, size=(512, 512))
    cut_face_lacking_scenes(video_output_path + "-real.avi", split_video_output_path, metadata, size=(512, 512))
