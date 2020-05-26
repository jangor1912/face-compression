import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from autoencoder.metric.metric import FaceMetric
from config.config import Config
from face_detection.face_landmarks import FaceLandmarksPredictorFAN
from face_detection.misc.image_helpers import ImageHelpers
from video.deconstructor.deconstructor import Deconstructor

CONF = Config().CONF


class ClipNotFormedError(Exception):
    def __init__(self,
                 *args,
                 video_path="",
                 expected_length=0,
                 start_frame=0,
                 end_frame=0,
                 clip_length=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_length = expected_length
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.clip_length = clip_length

    def __str__(self):
        return f"Clip of length {self.expected_length} cannot be formed from video {self.video_path}. " \
               f"Created clip began at frame {self.start_frame}, ended at frame {self.end_frame} " \
               f"and lasted {self.clip_length}"


class FaceTracker(object):
    def __init__(self, frame_rate=30, output_size=512, clip_number=2, clip_length=4):
        self.frame_rate = frame_rate
        self.output_size = output_size
        self.clip_number = clip_number
        self.clip_length = clip_length

    def video_to_clips(self, input_path, output_path):
        video_length = Deconstructor.get_video_length(input_path)
        frame_increase = int(video_length / self.clip_number)
        clip_number = 0
        start_frame = 0
        failure_number = 0
        while clip_number < self.clip_number:
            try:
                print(f"Started creating face clip at frame {start_frame}")
                face_clip, mask_clip = next(self.yield_face_clip(input_path, start_frame=start_frame))
                Deconstructor.save_video(output_path + f"-video{clip_number}.avi", face_clip,
                                         self.frame_rate, self.output_size)
                Deconstructor.save_video(output_path + f"-video{clip_number}-mask.avi", mask_clip,
                                         self.frame_rate, self.output_size)
                print(f"Successfully created face clip from frame {start_frame} "
                      f"to frame {start_frame + self.frame_rate * self.clip_length}")
                clip_number += 1
                failure_number = 0
                start_frame = clip_number * frame_increase
            except ClipNotFormedError as e:
                seconds_forward = np.power(2, failure_number)
                print(f"Forming clip starting at {start_frame} failed."
                      f" Moving cursor forward by {seconds_forward} second(s)")
                print(e)
                start_frame += self.frame_rate * seconds_forward
                failure_number += 1

    def yield_face_clip(self, input_path, start_frame=0):
        """Method that yields tuple (face_clip, mask_clip) if face could be tracked,
        else it raises ClipNotFormedError"""
        face_clip = []
        mask_clip = []
        current_frame = start_frame
        for item in self.yield_tracked_face(input_path, start_frame=start_frame):
            current_frame += 1
            # Face has been successfully tracked
            if item is not None:
                face, mask = item
                face_clip.append(face)
                mask_clip.append(mask)
                # Check if clip is long enough to return
                if len(face_clip) >= self.frame_rate * self.clip_length:
                    yield face_clip, mask_clip
                    face_clip = []
                    mask_clip = []
            else:
                raise ClipNotFormedError(start_frame=start_frame,
                                         end_frame=current_frame,
                                         clip_length=len(face_clip))

    def yield_tracked_face(self, input_path, start_frame=0):
        """Method that yields tuple (face_img, face_mask) if one exists else it yields None"""
        if not Path(input_path).exists():
            raise RuntimeError("Input path = {} does not exist!".format(input_path))
        frame_generator = Deconstructor.video_to_images(input_path, start_frame=start_frame)
        predictor = FaceLandmarksPredictorFAN(device="cuda")
        face_metric = FaceMetric(predictor)
        size = (self.output_size, self.output_size)

        # parse first frame
        frame, _ = next(frame_generator)
        img_height, img_width, _ = np.array(frame).shape

        face_location = None
        face_present = False
        prediction = predictor.detect_one_face_landmark(frame)
        if prediction is not None:
            face_location = predictor.get_face_square(frame, prediction)
            face_location = ImageHelpers.get_square(frame, face_location)
            face_location = ImageHelpers.expand_image(frame, face_location, expansion_factor=0.3)
            face_present = True

        last_goal = face_location or [int(img_height / 3), int(img_height / 3), int(img_height * 2 / 3),
                                      int(img_height * 2 / 3)]
        last_location = last_goal
        last_movement_speed = [0, 0]  # in pixels per frame
        last_resize_speed = 0  # in pixels per frame

        resize_smoothness = 4.0
        movement_smoothness = 4.0
        try:
            for frame, frame_no in frame_generator:
                img_height, img_width, _ = np.array(frame).shape

                prediction = predictor.detect_one_face_landmark(frame)
                image = frame
                mask = face_metric.generate_mask(prediction, img_height=img_height, img_width=img_width)
                if prediction is not None:
                    face_location = predictor.get_face_square(frame, prediction)
                    face_location = ImageHelpers.get_square(image, face_location)
                    face_location = ImageHelpers.expand_image(image, face_location, expansion_factor=0.3)
                    if not face_present:
                        last_resize_speed = 0
                        last_movement_speed = [0, 0]
                        last_location = face_location
                    face_present = True
                else:
                    face_location = last_goal
                    face_present = False
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
                image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

                mask = ImageHelpers.crop_image(mask, current_location)
                mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

                if prediction is None:
                    yield None
                else:
                    yield image, mask
        except Exception as e:
            print(str(e))
            raise


if __name__ == "__main__":
    # videos_dir = "/media/jan/Elements SE/Magisterka/youtube_dataset"
    # output_dir = "/media/jan/Elements SE/Magisterka/youtube_dataset/output"
    # current_video = "The Budget Phone Blueprint!.mp4"
    videos_dir = sys.argv[1]
    output_dir = sys.argv[2]
    current_video = sys.argv[3]
    video_path = os.path.join(videos_dir, current_video)
    video_output_path = os.path.join(output_dir, current_video.rstrip(".mp4") + "-processed")
    face_tracker = FaceTracker(clip_length=2, clip_number=4)
    face_tracker.video_to_clips(input_path=video_path,
                                output_path=video_output_path)
