import json
import os
import random
from os.path import isfile, join


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


if __name__ == "__main__":
    videos_path = "G:/Magisterka/kaggle_dataset/small/train_sample_videos"
    train = "G:/Magisterka/kaggle_dataset/small/train"
    test = "G:/Magisterka/kaggle_dataset/small/test"
    main(videos_path, train, test)
