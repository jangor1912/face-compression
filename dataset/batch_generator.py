import os
import random
from os.path import isfile, join
from time import time

import cv2
import numpy as np
from PIL import Image
from tensorflow.python.keras.utils import Sequence, OrderedEnqueuer

from video.deconstructor.deconstructor import Deconstructor


class BatchSequence(Sequence):

    def __init__(self, data_dir, input_size=(32, 32), batch_size=8, frames_no=8):
        self.data_dir = data_dir
        self.input_size = input_size
        self.frames_no = frames_no
        self.videos_paths = list()
        self.batch_size = batch_size
        self._init_directory()

        self.estimated_video_length = 1  # seconds
        self.video_frame_rate = 16  # fps
        self.video_length = self.estimated_video_length * self.video_frame_rate  # in frames

    def __len__(self):
        sequences_per_video = int(self.estimated_video_length * self.video_frame_rate / self.frames_no)
        result = int(np.floor(len(self.videos_paths) * sequences_per_video / self.batch_size))
        return result

    def __getitem__(self, index):
        start_video = index * self.batch_size
        start_frame = np.floor(start_video / len(self.videos_paths)) * self.frames_no
        start_video = start_video % len(self.videos_paths)
        if start_frame > self.video_length:
            raise RuntimeError("Trying to get frames out of video scope! "
                               "Video Length is {} frames, trying to access {} frame."
                               .format(self.video_length, start_frame))
        input_batch = list()
        target_batch = list()
        for i in range(start_video, start_video + self.batch_size):
            i = i % len(self.videos_paths)
            video_path, mask_path = self.videos_paths[i]
            video_seq, mask_seq = self.get_input(video_path, mask_path, int(start_frame))
            index_of_image_to_generate = random.randint(0, len(video_seq) - 1)
            input_batch.append(np.array(video_seq))
            target_batch.append([np.copy(video_seq[index_of_image_to_generate]),
                                 np.array(mask_seq[index_of_image_to_generate])])
        return np.array(input_batch), np.array(target_batch)

    def _init_directory(self):
        files = {join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))}
        for video_path in files:
            if video_path.endswith(".avi") and not video_path.endswith("-real.avi"):
                mask_video = video_path
                real_video = video_path[:-4] + "-real.avi"
                self.videos_paths.append((real_video, mask_video))
        random.shuffle(self.videos_paths)

    def get_input(self, video_path, mask_path, start_frame):
        video_sequence = list()
        mask_sequence = list()
        try:
            frame_generator = Deconstructor.video_to_images(video_path, start_frame=start_frame)
            mask_generator = Deconstructor.video_to_images(mask_path, start_frame=start_frame)
            for (frame, frame_no), (mask, mask_no) in zip(frame_generator, mask_generator):
                if frame_no >= self.frames_no:
                    break
                frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_AREA)
                frame = np.array(frame, dtype=np.float32)
                mask = np.array(mask, dtype=np.float32)
                frame = frame / 255
                frame = frame - 0.5
                mask = mask / 255
                mask = mask + 1
                video_sequence.append(frame)
                mask_sequence.append(mask)
            if len(video_sequence) != self.frames_no:
                raise RuntimeError("Sequence of {} frames cannot be read from video {}. Already read {} frames."
                                   .format(self.frames_no, video_path, len(video_sequence)))
        except Exception:
            raise
        return np.array(video_sequence), np.array(mask_sequence)


def test(input_dir, output_dir):
    sequence = BatchSequence(input_dir, input_size=(32, 32),
                             batch_size=4, frames_no=8)
    enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=True, shuffle=True)
    print("Length of sequence is {}.".format(len(sequence)))
    start_time = time()
    enqueuer.start(workers=4, max_queue_size=24)
    generator = enqueuer.get()
    img_no = 0
    for _ in range(10):
        input_batch, target_batch = generator.next()
        for input_seq, target in zip(list(input_batch), list(target_batch)):
            for input_img in list(input_seq):
                target_img, target_mask = list(target)

                input_img = input_img + 0.5
                input_img = input_img * 255
                input_img = np.array(input_img, dtype=np.uint8)
                target_img = target_img + 0.5
                target_img = target_img * 255
                target_img = np.array(target_img, dtype=np.uint8)
                target_mask = target_mask - 1
                target_mask = target_mask * 255
                target_mask = np.array(target_mask, dtype=np.uint8)

                input_img = Image.fromarray(input_img)
                target_img = Image.fromarray(target_img)
                target_mask = Image.fromarray(target_mask)
                input_img.save(os.path.join(output_dir, "input_img_" + str(img_no) + ".jpg"))
                target_img.save(os.path.join(output_dir, "target_img_" + str(img_no) + ".jpg"))
                target_mask.save(os.path.join(output_dir, "target_mask_" + str(img_no) + ".jpg"))
                img_no += 1
    end_time = time()
    print("Function took {} s".format(end_time - start_time))


if __name__ == "__main__":
    test("/media/jan/Elements SE/Magisterka/kaggle_dataset/small/train/final",
         "/media/jan/Elements SE/Magisterka/temporary")
