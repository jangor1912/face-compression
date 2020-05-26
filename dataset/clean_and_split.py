import json
import os
import random
import re
from os.path import isfile, join
from pathlib import Path


def main(path, train_path, test_path):
    cwd = path
    files = [f for f in os.listdir(cwd) if isfile(join(cwd, f))]
    print(f"Files = {files}")
    with open(join(cwd, "metadata.json")) as json_file:
        files_dict = json.load(json_file)

    video_files = []
    print(f"Files_dict = {files_dict}")
    for file in files:
        if file in files_dict:
            if files_dict[file]["label"] == "FAKE":
                os.remove(join(cwd, file))
            else:
                video_files.append(file)

    random.shuffle(video_files)
    videos_no = len(video_files)
    split = int(videos_no * 0.7)
    train_files = video_files[:split]
    test_files = video_files[split:]

    for file in train_files:
        os.rename(join(cwd, file), join(train_path, file))

    for file in test_files:
        os.rename(join(cwd, file), join(test_path, file))


def get_video_mask_tuples(videos_directory):
    result = []
    files = os.listdir(videos_directory)
    for video_path in files:
        if re.search(r'.*-processed-video\d.avi', video_path):
            org_video_path = video_path[:video_path.find(".avi")]
            mask_video = org_video_path + "-mask.avi"
            result.append((video_path, mask_video))
    return result


def populate_directories(videos_dir, train_dir, test_dir):
    all_videos = get_video_mask_tuples(videos_dir)
    random.shuffle(all_videos)
    videos_no = len(all_videos)
    split = int(videos_no * 0.7)
    train_files = all_videos[:split]
    test_files = all_videos[split:]

    for video, mask in train_files:
        try:
            os.rename(join(videos_dir, video), join(train_dir, video))
            os.rename(join(videos_dir, mask), join(train_dir, mask))
        except Exception as e:
            print(f"Error while processing train video {video}. Error = {str(e)}")

    for video, mask in test_files:
        try:
            os.rename(join(videos_dir, video), join(test_dir, video))
            os.rename(join(videos_dir, mask), join(test_dir, mask))
        except Exception as e:
            print(f"Error while processing test video {video}. Error = {str(e)}")


if __name__ == "__main__":
    videos_path = Path("G:/Magisterka/youtube_dataset/output/all")
    train = Path("G:/Magisterka/youtube_dataset/output/train")
    test = Path("G:/Magisterka/youtube_dataset/output/test")
    populate_directories(videos_path, train, test)
