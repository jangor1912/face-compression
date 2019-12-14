import time
from pathlib import Path

import cv2
import sys

from face_detection.face_landmarks import place_landmarks
from video.deconstructor.deconstructor import Deconstructor


def show_image(img, show_time=1):
    cv2.imshow("Output", img)
    time.sleep(show_time)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # video_path = "C:\\Users\\Scarf_000\\Documents\\Python Scripts\\Magisterka\\Facial-landmarks\\head_movement.mp4"
    video_path = "G:\\Ubuntu_shared\\Face-compression\\Data\\Video\\Billie Eilish Same Interview Ten Minutes Apart.mp4"
    # output_path = video_path.rstrip(".mp4") + "-landmarks.avi"
    output_path = "C:\\Users\\Scarf_000\\Documents\\Python Scripts\\Magisterka\\Facial-landmarks\\head_movement_lol.mp4"
    if Path(video_path).exists():
        print("Output path = {} exists!".format(video_path))
    else:
        print("Output path = {} does not exist!".format(video_path))
        sys.exit()

    frame_generator = Deconstructor().video_to_images(video_path)
    first_frame, _ = frame_generator.__next__()
    height, width, layers = first_frame.shape
    size = (width, height)
    # video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for frame, frame_no in frame_generator:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Output", gray)
        # image = place_landmarks(frame, image_path=False)
        #
        # video_writer.write(image)
    cv2.destroyAllWindows()
    # video_writer.release()
