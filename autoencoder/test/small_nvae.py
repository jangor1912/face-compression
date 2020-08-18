from autoencoder.models.small_nvae import NVAEAutoEncoder64
from autoencoder.train.train import get_default_hparams
from callbacks.callbacks import DotDict
from dataset.batch_generator import NVAESequence
from video.deconstructor.deconstructor import Deconstructor


class Testing(object):
    def __init__(self, encoder_frames_no,
                 test_directory,
                 checkpoint_path=None):
        self.encoder_frames_no = encoder_frames_no
        self.checkpoint_path = checkpoint_path
        self.auto_encoder = self.get_auto_encoder()
        self.face_encoding = None
        self.sequence = NVAESequence(test_directory,
                                     input_size=(64, 64),
                                     batch_size=1,
                                     encoder_frames_no=encoder_frames_no)

    def get_auto_encoder(self):
        model_params = DotDict(get_default_hparams())
        auto_encoder = NVAEAutoEncoder64(model_params,
                                         batch_size=1,
                                         encoder_frames_no=self.encoder_frames_no)
        auto_encoder.summary()
        metrics = [auto_encoder.loss_func,
                   auto_encoder.face_metric,
                   auto_encoder.face_kl_loss,
                   auto_encoder.mask_mse_loss,
                   "mae", "mse"]
        auto_encoder.model.compile(loss=auto_encoder.loss_func,
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
        encoder_seq.reshape((1, self.encoder_frames_no, 64, 64, 3))
        encoded_face = self.auto_encoder.encoder_model.predict(encoder_seq,
                                                               batch_size=1)
        return encoded_face

    def decode_video(self, face_video_path, mask_video_path):
        last_frame = Deconstructor.get_video_length(face_video_path)
        video_seq, mask_seq = self.sequence.get_input(face_video_path, mask_video_path, 1,
                                                      frames_no=last_frame - 1)
        # TODO Finish it!
        return
