import os
import random
import re
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

        self.estimated_video_length = 1.5  # seconds
        self.video_frame_rate = 30  # fps
        self.video_length = self.estimated_video_length * self.video_frame_rate  # in frames
        self._check_data_set()

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
        video_batch = list()
        mask_batch = list()
        target_batch = list()
        for i in range(start_video, start_video + self.batch_size):
            i = i % len(self.videos_paths)
            video_path, mask_path = self.videos_paths[i]
            video_seq, mask_seq = self.get_input(video_path, mask_path, int(start_frame))
            index_of_image_to_generate = random.randint(0, len(video_seq) - 1)
            video_batch.append(np.array(video_seq))
            input_mask = np.copy(mask_seq[index_of_image_to_generate])
            mask_batch.append(input_mask)
            target_mask = np.copy(input_mask)
            target_mask = self.np_mask_to_rgb_image(target_mask)
            target_mask = self.rgb_image_to_np_array(target_mask)
            target_batch.append(
                [np.copy(video_seq[index_of_image_to_generate]),
                 target_mask])
        return [np.array(video_batch), np.array(mask_batch)], np.array(target_batch)

    def generate(self):
        while True:
            for i in range(len(self)):
                yield self[i]

    def _init_directory(self):
        files = {join(self.data_dir, f) for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))}
        for video_path in files:
            if re.search(r'.*-processed-video\d.avi', video_path):
                org_video_path = video_path[:video_path.find(".avi")]
                mask_video = org_video_path + "-mask.avi"
                self.videos_paths.append((video_path, mask_video))
        random.shuffle(self.videos_paths)

    def _check_data_set(self):
        for video_path, mask_path in self.videos_paths:
            video_length = Deconstructor.get_video_length(video_path)
            mask_length = Deconstructor.get_video_length(mask_path)
            if video_length <= self.video_length:
                raise RuntimeError(f"Video {video_path} is shorter than estimated video length")
            if mask_length <= self.video_length:
                raise RuntimeError(f"Video {mask_path} is shorter than estimated video length")

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
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = np.array(mask, dtype=np.float32)
                mask = np.expand_dims(mask, axis=-1)
                frame = self.rgb_image_to_np_array(frame)
                mask = self.rgb_image_to_np_array(mask)
                video_sequence.append(frame)
                mask_sequence.append(mask)
            if len(video_sequence) != self.frames_no:
                raise RuntimeError("Sequence of {} frames cannot be read from video {}. Already read {} frames."
                                   .format(self.frames_no, video_path, len(video_sequence)))
        except Exception:
            raise
        return np.array(video_sequence), np.array(mask_sequence)

    @classmethod
    def np_mask_to_rgb_image(cls, np_mask):
        np_mask = np.squeeze(np_mask, axis=-1)
        np_mask = np_mask + 1
        np_mask = np_mask * (255/2)
        np_mask = np.uint8(np_mask)
        np_mask = cv2.cvtColor(np_mask, cv2.COLOR_GRAY2RGB)
        mask_img = Image.fromarray(np_mask, mode="RGB")
        return mask_img

    @classmethod
    def np_img_to_rgb_image(cls, np_image):
        np_image = np_image + 1
        np_image = np_image * (255/2)
        np_image = np.uint8(np_image)
        rgb_image = Image.fromarray(np_image, mode="RGB")
        return rgb_image

    @classmethod
    def rgb_image_to_np_array(cls, rgb_img):
        rgb_img = np.array(rgb_img, dtype=np.float32)
        rgb_img *= 2/255
        rgb_img -= 1
        return rgb_img


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
        input_batch, target_batch = next(generator)
        input_seq_batch = input_batch[0]
        input_mask_batch = input_batch[1]
        for input_seq, input_mask, _target in zip(list(input_seq_batch), list(input_mask_batch), list(target_batch)):
            target_img = _target[0]
            target_mask = _target[1]
            target_img = BatchSequence.np_img_to_rgb_image(target_img)
            target_mask = BatchSequence.np_img_to_rgb_image(target_mask)
            target_img.save(os.path.join(output_dir, "target_img_" + str(img_no) + ".jpg"))
            target_mask.save(os.path.join(output_dir, "target_mask_" + str(img_no) + ".jpg"))
            for input_img in list(input_seq):
                input_img = BatchSequence.np_img_to_rgb_image(input_img)
                input_img.save(os.path.join(output_dir, "input_img_" + str(img_no) + ".jpg"))
                img_no += 1
    end_time = time()
    print("Function took {} s".format(end_time - start_time))


if __name__ == "__main__":
    test("G:/Magisterka/kaggle_dataset/small/train/final",
         "G:/Magisterka/temporary")
