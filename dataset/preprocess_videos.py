import os
from os.path import isfile, join

from video.video_to_landmarks import generate_landmarked_face_video, cut_face_lacking_scenes


def main(videos_dir, final_dir):
    files = [f for f in os.listdir(videos_dir) if isfile(join(videos_dir, f))]

    for file in files:
        try:
            print(f"Started processing video {file}")
            file_name = file.rstrip(".mp4")
            metadata = generate_landmarked_face_video(join(videos_dir, file), join(final_dir, file_name),
                                                      length=4, strict=True)
            output_file = join(final_dir, file_name)
            split_video_output_path = output_file + "-split"
            split_mask_video_output_path = output_file + "-split-mask"
            cut_face_lacking_scenes(output_file + ".avi", split_mask_video_output_path, metadata)
            cut_face_lacking_scenes(output_file + "-real.avi", split_video_output_path, metadata)
            print(f"Successfully processed video {file}")
        except Exception as e:
            print(f"Error during processing {file}, error is {str(e)}")


if __name__ == "__main__":
    videos_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test"
    final_directory = "/media/jan/Elements SE/Magisterka/kaggle_dataset/small/test/final2"
    main(videos_directory, final_directory)
