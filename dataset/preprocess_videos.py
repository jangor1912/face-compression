import os
from os.path import isfile, join

from video_to_landmarks import generate_landmarked_face_video


def main(videos_dir, final_dir):
    files = [f for f in os.listdir(videos_dir) if isfile(join(videos_dir, f))]

    for file in files:
        try:
            print(f"Started processing video {file}")
            generate_landmarked_face_video(join(videos_dir, file), join(final_dir, file.rstrip(".mp4")))
            print(f"Successfully processed video {file}")
        except Exception as e:
            print(f"Error during processing {file}, error is {str(e)}")


if __name__ == "__main__":
    videos_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test"
    final_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test/final"
    main(videos_directory, final_directory)
