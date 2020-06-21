import os
from pathlib import Path

from video.face_tracker import FaceTracker


def process_videos(videos_dir, output_dir,
                   clip_length=2, clip_number=4):
    face_tracker = FaceTracker(clip_length=clip_length,
                               clip_number=clip_number,
                               strict=False)
    files = [f for f in os.listdir(videos_dir)
             if Path(videos_dir, f).is_file()]

    for file in files:
        try:
            print(f"Started processing video {file}")
            face_tracker.video_to_clips(input_path=str(Path(videos_dir, file)),
                                        output_path=os.path.splitext(Path(output_dir, file))[0])
            print(f"Successfully processed video {file}")
        except Exception as e:
            print(f"Error during processing {file}, error is {str(e)}")
            raise e


if __name__ == "__main__":
    videos_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test"
    final_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test/final2"
    process_videos(videos_directory, final_directory)
