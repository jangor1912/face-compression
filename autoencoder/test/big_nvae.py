import cv2
import numpy as np
from tensorflow.python.keras.optimizers import Adamax

from autoencoder.models.big_nvae import BigNVAEAutoEncoder128
from autoencoder.train.train import get_default_hparams
from callbacks.callbacks import DotDict
from dataset.batch_generator import NVAESequence
from video.deconstructor.deconstructor import Deconstructor


class BigNvaeDecoder(object):
    def __init__(self, encoder_frames_no,
                 test_directory,
                 checkpoint_path=None):
        self.encoder_frames_no = encoder_frames_no
        self.checkpoint_path = checkpoint_path
        self.auto_encoder = self.get_auto_encoder()
        self.face_encoding = None
        self.input_size = (128, 128)
        self.sequence = NVAESequence(test_directory,
                                     input_size=self.input_size,
                                     batch_size=1,
                                     encoder_frames_no=encoder_frames_no)

    def get_auto_encoder(self):
        model_params = DotDict(get_default_hparams())
        auto_encoder = BigNVAEAutoEncoder128(model_params,
                                             batch_size=1,
                                             encoder_frames_no=self.encoder_frames_no)
        auto_encoder.summary()
        metrics = [auto_encoder.loss_func,
                   auto_encoder.face_metric,
                   auto_encoder.face_kl_loss,
                   auto_encoder.mask_mse_loss,
                   "mae", "mse"]
        optimizer = Adamax(auto_encoder.hps.learning_rate)
        auto_encoder.model.compile(loss=auto_encoder.loss_func,
                                   optimizer=optimizer,
                                   metrics=metrics)
        if self.checkpoint_path:
            # Load weights:
            load_status = auto_encoder.model.load_weights(self.checkpoint_path)
            if load_status:
                load_status.assert_consumed()
        return auto_encoder

    def encode_face(self, face_video_path, mask_video_path):
        encoder_seq, _ = self.sequence.get_input(face_video_path, mask_video_path, 0,
                                                 frames_no=self.encoder_frames_no)
        encoder_seq = encoder_seq.reshape((1, self.encoder_frames_no, 128, 128, 3))
        encoded_face = self.auto_encoder.encoder_model.predict(encoder_seq,
                                                               batch_size=1)
        return encoded_face

    def decode_video(self, video_path, mask_path,
                     encoded_face, output_video_path):
        start_frame = 0
        last_frame = Deconstructor.get_video_length(video_path)
        frames_no = last_frame - 1

        video_writer = cv2.VideoWriter(output_video_path,
                                       cv2.VideoWriter_fourcc(*"XVID"),
                                       30, (128, 128))

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
            frame = self.sequence.rgb_image_to_np_array(frame)
            mask = self.sequence.rgb_image_to_np_array(mask)

            # change to batch
            frame = frame.reshape((1, 128, 128, 3))
            mask = mask.reshape((1, 128, 128, 1))

            # predict frame output with model
            encoded_mask = self.auto_encoder.mask_encoder_model.predict(mask,
                                                                        batch_size=1)
            decoded_frame = self.auto_encoder.decoder_model.predict(encoded_face + encoded_mask)

            # save to video file
            decoded_frame = decoded_frame.reshape((128, 128, 3))
            decoded_frame += 1.0
            decoded_frame *= 255.0 / 2.0
            decoded_frame = np.uint8(decoded_frame)
            video_writer.write(decoded_frame)

            if frame_no % 30 == 0:
                print(f"Processed {round((frame_no / last_frame * 100), 2)}% of video")

        video_writer.release()
