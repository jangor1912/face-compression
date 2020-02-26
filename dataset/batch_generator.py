import os
import random
from os.path import isfile, join
from pprint import PrettyPrinter

import cv2
import numpy as np

from video.deconstructor.deconstructor import Deconstructor


class BatchGenerator(object):
    def __init__(self, data_dir, input_size=(32, 32), batch_size=8, frames_no=8):
        self.data_dir = data_dir
        self.input_size = input_size
        self.frames_no = frames_no
        self.processed_frames = dict()
        self.processed_videos = set()
        self.videos_paths = list()
        self.current_video = 0
        self.batch_size = batch_size
        self._init_directory()

        self.estimated_video_length = 4  # seconds
        self.video_frame_rate = 30  # fps
        self.video_length = self.estimated_video_length * self.video_frame_rate  # in frames

    def __len__(self):
        sequences_per_video = int(self.estimated_video_length * self.video_frame_rate / self.frames_no)
        return int(np.floor(len(self.videos_paths) * sequences_per_video / self.batch_size))

    def _init_directory(self):
        files = {join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))}
        for video_path in files:
            if video_path.endswith(".avi") and not video_path.endswith("-real.avi"):
                mask_video = video_path
                real_video = video_path.rstrip(".avi") + "-real.avi"
                self.videos_paths.append((real_video, mask_video))

    def get_input(self, video_path, mask_path, start_frame):
        video_sequence = list()
        mask_sequence = list()
        try:
            frame_generator = Deconstructor().video_to_images(video_path, start_frame=start_frame)
            mask_generator = Deconstructor().video_to_images(mask_path, start_frame=start_frame)
            for (frame, frame_no), (mask, mask_no) in zip(frame_generator, mask_generator):
                if frame_no >= self.frames_no:
                    break
                frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_AREA)
                frame = np.array(frame)
                mask = np.array(mask)
                frame = frame / 255
                frame = frame - 0.5
                mask = mask / 255
                mask = mask + 1
                video_sequence.append(frame)
                mask_sequence.append(mask)
            if len(video_sequence) != self.frames_no:
                self.processed_videos.add(video_path)
                raise RuntimeError(f"Sequence of {self.frames_no} frames cannot be read from video {video_path}")
            self.processed_frames[video_path] += len(video_sequence)
        except Exception:
            self.processed_videos.add(video_path)
            raise
        return np.array(video_sequence), np.array(mask_sequence)

    def _get_item(self, index):
        video_seq = list()
        mask_seq = list()
        index = index % len(self.videos_paths)
        current_video = None
        while current_video is None:
            current_video, current_mask = self.videos_paths[index]
            try:
                if current_video not in self.processed_frames:
                    self.processed_frames[current_video] = 0
                if current_video in self.processed_videos:
                    raise RuntimeError(f"Video {current_video} was already processed!")
                if self.processed_frames[current_video] + self.frames_no > self.video_length:
                    self.processed_videos.add(current_video)
                    raise RuntimeError(f"Successfully processed video {current_video}")
                video_seq, mask_seq = self.get_input(current_video, current_mask,
                                                     self.processed_frames[current_video])
            except Exception as e:
                print(e)
                current_video = None
                index += 1
                index = index % len(self.videos_paths)
                pass
        self.current_video = index
        self.current_video += 1
        self.current_video = self.current_video % len(self.videos_paths)
        index_of_image_to_generate = random.randint(0, len(video_seq) - 1)
        return np.array(video_seq), [np.copy(video_seq[index_of_image_to_generate]),
                                     np.array(mask_seq[index_of_image_to_generate])]

    def get_item(self):
        try:
            inputs, targets = self._get_item(self.current_video)
        except Exception as e:
            print("Error during getting batch!")
            print(e)
            print("Resetting processed videos!")
            self.processed_videos = set()
            self.processed_frames = dict()
            random.shuffle(self.videos_paths)
            inputs, targets = self._get_item(self.current_video)
        return inputs, targets

    def get_batch(self):
        while True:
            input_batch = list()
            target_batch = list()
            for _ in range(self.batch_size):
                inputs, targets = self.get_item()
                input_batch.append(inputs)
                target_batch.append(targets)
            yield np.array(input_batch), np.array(target_batch)


def test(directory):
    generator = BatchGenerator(directory, input_size=(32, 32),
                               batch_size=4, frames_no=8)
    pp = PrettyPrinter(indent=4)
    for batch in generator.get_batch():
        pp.pprint(generator.processed_frames)
        pp.pprint(generator.processed_videos)


if __name__ == "__main__":
    test("/media/jan/Elements SE/Magisterka/kaggle_dataset/small/train/final")
