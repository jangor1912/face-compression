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

        self.estimated_video_length = 1.9  # seconds
        self.video_frame_rate = 30  # fps
        self.video_length = self.estimated_video_length * self.video_frame_rate  # in frames
        print(f"Started checking dataset in folder {self.data_dir}")
        # self._check_data_set()
        print(f"Finished checking dataset in folder {self.data_dir}")

    def __len__(self):
        sequences_per_video = int(self.estimated_video_length * self.video_frame_rate / float(self.frames_no))
        result = int(np.floor(len(self.videos_paths) * sequences_per_video / float(self.batch_size)))
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
        detail_batch = list()
        for i in range(start_video, start_video + self.batch_size):
            i = i % len(self.videos_paths)
            video_path, mask_path = self.videos_paths[i]
            video_seq, mask_seq = self.get_input(video_path, mask_path, int(start_frame))
            index_of_image_to_generate = random.randint(0, len(video_seq) - 1)
            index_of_detail_image = random.randint(0, Deconstructor.get_video_length(video_path) - 1 - self.frames_no)
            detail_seq, _ = self.get_input(video_path, mask_path, int(index_of_detail_image))
            detail_batch.append(np.array(detail_seq[0]))
            video_batch.append(np.array(video_seq))
            input_mask = np.copy(mask_seq[index_of_image_to_generate])
            mask_batch.append(input_mask)
            target_mask = np.copy(input_mask)
            target_mask = self.np_mask_to_rgb_image(target_mask)
            target_mask = self.rgb_image_to_np_array(target_mask)
            target_batch.append(
                [np.copy(video_seq[index_of_image_to_generate]),
                 target_mask])
        return [np.array(video_batch), np.array(detail_batch), np.array(mask_batch)], np.array(target_batch)

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
        for i, video_tuple in enumerate(self.videos_paths):
            video_path, mask_path = video_tuple
            print(f"Checking face video {video_path}")
            video_length = Deconstructor.get_video_length(video_path)
            print(f"Checking mask video {mask_path}")
            mask_length = Deconstructor.get_video_length(mask_path)
            if video_length <= self.video_length:
                raise RuntimeError(f"Video {video_path} is shorter than estimated video length")
            if mask_length <= self.video_length:
                raise RuntimeError(f"Video {mask_path} is shorter than estimated video length")
            print(f"Checked {i}/{len(self.videos_paths)}")

    def get_input(self, video_path, mask_path, start_frame, frames_no=None):
        frames_no = frames_no or self.frames_no
        video_sequence = list()
        mask_sequence = list()
        try:
            frame_generator = Deconstructor.video_to_images(video_path, start_frame=start_frame)
            mask_generator = Deconstructor.video_to_images(mask_path, start_frame=start_frame)
            for (frame, frame_no), (mask, mask_no) in zip(frame_generator, mask_generator):
                if frame_no >= start_frame + frames_no:
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
            if len(video_sequence) != frames_no:
                raise RuntimeError(f"Sequence of {frames_no} frames cannot be read from video {video_path}, "
                                   f"starting at frame {start_frame}. Already read {len(video_sequence)} frames.")
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
        np_image += 1.0
        np_image *= 255.0/2.0
        np_image = np.uint8(np_image)
        bgr_img = Image.fromarray(np_image, mode="RGB")
        b, g, r = bgr_img.split()
        final_img = Image.merge("RGB", (r, g, b))
        return final_img

    @classmethod
    def rgb_image_to_np_array(cls, rgb_img):
        rgb_img = np.array(rgb_img, dtype=np.float32)
        rgb_img *= 2.0/255.0
        rgb_img -= 1.0
        return rgb_img


class LSTMSequence(BatchSequence):
    def __init__(self, data_dir, input_size=(128, 128), batch_size=8,
                 frames_no=3, encoder_frames_no=30):
        super().__init__(data_dir, input_size, batch_size, frames_no)
        self.encoder_frames_no = encoder_frames_no
        self.last_possible_encoder_frame = self.video_length - self.encoder_frames_no - 1
        assert self.last_possible_encoder_frame > 0, f"Cannot properly encode {self.encoder_frames_no} frames " \
                                                     f"using videos only {self.video_length} (frames) long!"

    def __len__(self):
        sequences_per_video = 1
        result = int(np.floor(len(self.videos_paths) * sequences_per_video / float(self.batch_size)))
        return result

    def __getitem__(self, index):
        start_video = index * self.batch_size
        start_frame = np.floor(start_video / len(self.videos_paths)) * self.frames_no
        start_video = start_video % len(self.videos_paths)
        if start_frame > self.video_length:
            raise RuntimeError("Trying to get frames out of video scope! "
                               "Video Length is {} frames, trying to access {} frame."
                               .format(self.video_length, start_frame))
        encoder_batch = list()
        mask_batch = list()
        target_batch = list()
        detail_batch = list()
        for i in range(start_video, start_video + self.batch_size):
            i = i % len(self.videos_paths)
            video_path, mask_path = self.videos_paths[i]
            encoder_start = random.randint(0, self.last_possible_encoder_frame)
            encoder_seq, _ = self.get_input(video_path, mask_path, int(encoder_start),
                                            frames_no=self.encoder_frames_no)
            encoder_batch.append(np.array(encoder_seq))

            index_of_image_to_generate = random.randint(self.frames_no + 1,
                                                        Deconstructor.get_video_length(video_path) - self.frames_no - 1)
            video_seq, mask_seq = self.get_input(video_path, mask_path,
                                                 start_frame=index_of_image_to_generate - self.frames_no,
                                                 frames_no=self.frames_no)
            index_of_detail_image = random.randint(0, Deconstructor.get_video_length(video_path) - 1)
            detail_seq, _ = self.get_input(video_path, mask_path, int(index_of_detail_image), frames_no=1)
            detail_batch.append(np.array(detail_seq[0]))
            input_mask_seq = np.array(mask_seq)
            mask_batch.append(input_mask_seq)
            target_mask = np.copy(input_mask_seq[-1])
            target_mask = self.np_mask_to_rgb_image(target_mask)
            target_mask = self.rgb_image_to_np_array(target_mask)
            target_batch.append(
                [np.copy(video_seq[-1]),
                 target_mask])
        return [np.array(encoder_batch), np.array(detail_batch), np.array(mask_batch)], np.array(target_batch)


def test(input_dir, output_dir):
    sequence = BatchSequence(input_dir, input_size=(32, 32),
                             batch_size=4, frames_no=8)
    enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=True, shuffle=True)
    print("Length of sequence is {}.".format(len(sequence)))
    start_time = time()
    enqueuer.start(workers=4, max_queue_size=128)
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
    test("G:/Magisterka/youtube_dataset/output/train",
         "G:/Magisterka/temporary")
