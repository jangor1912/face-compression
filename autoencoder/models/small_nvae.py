import numpy as np
import tensorflow.python.keras.backend as k
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, ZeroPadding2D, TimeDistributed

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.blocks import encoder_convolutional_block, encoder_residual_block, \
    mask_residual_block, mobnet_conv_block, mobnet_separable_conv_block, decoder_convolutional_block
from autoencoder.models.utils import SimpleSamplingLayer, DummyMaskLayer


class Architecture(object):
    """
    generic architecture template
    """

    def __init__(self, input_shape=None, batch_size=None, frames_no=None):
        """
        params:
        ---------
        input_shape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        """
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.frames_no = frames_no
        self.real_input_shape = (self.frames_no,) + self.input_shape
        self.dropout = 0.4
        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class MaskEncoder64(Architecture):
    def __init__(self, hps, batch_size=16, frames_no=30):
        self.hps = hps
        self.mask32x32x64 = None
        self.mask16x16x128 = None
        self.mask8x8x256 = None
        self.mask4x4x512 = None
        super(MaskEncoder64, self).__init__(input_shape=(64, 64, 1),
                                            batch_size=batch_size,
                                            frames_no=frames_no)

    def layers(self):
        mask_input = Input(self.input_shape, self.batch_size, name="mask_input")
        # 64x64x3
        x = mobnet_conv_block(mask_input, num_filters=32, kernel_size=3, strides=2)
        # 32x32x32
        x = mobnet_separable_conv_block(x, num_filters=32, strides=1)
        x = mobnet_conv_block(x, num_filters=64, kernel_size=1)
        # 32x32x64
        self.mask32x32x64 = mask_residual_block(x, depth=64)
        x = mobnet_separable_conv_block(x, num_filters=64, strides=2)
        # 16x16x64
        x = mobnet_conv_block(x, num_filters=128, kernel_size=1)
        # 16x16x128
        self.mask16x16x128 = mask_residual_block(x, depth=128)
        x = mobnet_separable_conv_block(x, num_filters=128, strides=1)
        x = mobnet_conv_block(x, num_filters=128, kernel_size=1)
        x = mobnet_separable_conv_block(x, num_filters=128, strides=2)
        # 8x8x128
        x = mobnet_conv_block(x, num_filters=256, kernel_size=1)
        # 8x8x256
        self.mask8x8x256 = mask_residual_block(x, depth=256)
        x = mobnet_separable_conv_block(x, num_filters=256, strides=1)
        x = mobnet_conv_block(x, num_filters=256, kernel_size=1)
        x = mobnet_separable_conv_block(x, num_filters=256, strides=2)
        # 4x4x256
        x = mobnet_conv_block(x, num_filters=512, kernel_size=1)
        # 4x4x512
        self.mask4x4x512 = mask_residual_block(x, depth=512)

        return mask_input, [self.mask4x4x512,
                            self.mask8x8x256,
                            self.mask16x16x128,
                            self.mask32x32x64]

    def Build(self):
        encoder_input, encoder_output = self.layers()
        return Model(inputs=encoder_input, outputs=encoder_output)


class NVAEEncoder64(Architecture):
    def __init__(self, hps, batch_size=16, frames_no=30):
        self.hps = hps

        self.mean_32x32x64 = None
        self.stddev_32x32x64 = None

        self.mean_16x16x128 = None
        self.stddev_16x16x128 = None

        self.mean_8x8x256 = None
        self.stddev_8x8x256 = None

        self.mean_4x4x512 = None
        self.stddev_4x4x512 = None

        super(NVAEEncoder64, self).__init__(input_shape=(64, 64, 3),
                                            batch_size=batch_size,
                                            frames_no=frames_no)

    def layers(self):
        input_layer = Input(self.real_input_shape, self.batch_size)
        # 64x64x3
        # Zero-Padding
        net = TimeDistributed(ZeroPadding2D((3, 3)))(input_layer)

        # 70x70x3

        # Stage 0
        net = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), name='conv0',
                                     kernel_initializer=glorot_uniform(seed=0)))(net)
        net = BatchNormalization(axis=-1, name='bn_conv0')(net)
        net = Activation('relu')(net)

        # 32x32x64

        # Stage 1
        net = encoder_convolutional_block(net, f=3, filters=[64, 64, 128], stage=1, block='a', s=1)
        # net = encoder_identity_block(net, 3, [64, 64, 128], stage=1, block='b')

        # 32x32x64
        self.mean_32x32x64 = encoder_residual_block(net, depth=64)
        self.stddev_32x32x64 = encoder_residual_block(net, depth=64)

        # Stage 2
        net = encoder_convolutional_block(net, f=3, filters=[64, 64, 128], stage=2, block='a', s=2)
        # net = encoder_identity_block(net, 3, [64, 64, 128], stage=2, block='b')

        # 16x16x128
        self.mean_16x16x128 = encoder_residual_block(net, depth=128)
        self.stddev_16x16x128 = encoder_residual_block(net, depth=128)

        # Stage 3
        net = encoder_convolutional_block(net, f=3, filters=[128, 128, 256], stage=3, block='a', s=2)
        # net = encoder_identity_block(net, 3, [128, 128, 256], stage=3, block='b')

        # 8x8x256
        self.mean_8x8x256 = encoder_residual_block(net, depth=256)
        self.stddev_8x8x256 = encoder_residual_block(net, depth=256)

        # Stage 4
        net = encoder_convolutional_block(net, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
        # net = encoder_identity_block(net, 3, [256, 256, 512], stage=4, block='b')

        # 4x4x512
        self.mean_4x4x512 = encoder_residual_block(net, depth=512)
        self.stddev_4x4x512 = encoder_residual_block(net, depth=512)

        return input_layer, [self.mean_4x4x512, self.stddev_4x4x512,
                             self.mean_8x8x256, self.stddev_8x8x256,
                             self.mean_16x16x128, self.stddev_16x16x128,
                             self.mean_32x32x64, self.stddev_32x32x64]

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class NVAEDecoder64(Architecture):
    def __init__(self, hps, batch_size=4):
        self.hps = hps

        super(NVAEDecoder64, self).__init__(input_shape=(64, 64, 3),
                                            batch_size=batch_size)

    def layers(self):
        # Face encoder inputs
        mean_4x4x512 = Input((4, 4, 512), self.batch_size, name="mean_4x4x512")
        stddev_4x4x512 = Input((4, 4, 512), self.batch_size, name="stddev_4x4x512")
        mean_8x8x256 = Input((8, 8, 256), self.batch_size, name="mean_8x8x256")
        stddev_8x8x256 = Input((8, 8, 256), self.batch_size, name="stddev_8x8x256")
        mean_16x16x128 = Input((16, 16, 128), self.batch_size, name="mean_16x16x128")
        stddev_16x16x128 = Input((16, 16, 128), self.batch_size, name="stddev_16x16x128")
        mean_32x32x64 = Input((32, 32, 64), self.batch_size, name="mean_32x32x64")
        stddev_32x32x64 = Input((32, 32, 64), self.batch_size, name="stddev_32x32x64")

        # Mask encoder inputs
        mask4x4x512 = Input((4, 4, 512), self.batch_size, name="mask4x4x512")
        mask8x8x256 = Input((8, 8, 256), self.batch_size, name="mask8x8x256")
        mask16x16x128 = Input((16, 16, 128), self.batch_size, name="mask16x16x128")
        mask32x32x64 = Input((32, 32, 64), self.batch_size, name="mask32x32x64")

        ###################
        # Decoder network #
        ###################

        # sample2x2x1024 = SimpleSamplingLayer()([mean_2x2x1024, stddev_2x2x1024, mask2x2x1024])
        # net = sample2x2x1024
        #
        # # Stage 6
        # net = decoder_convolutional_block(net, f=3, filters=[1024, 1024, 512], stage=6, block='a', s=2)
        # # net = decoder_convolutional_block(net, f=3, filters=[1024, 1024, 512], stage=6, block='b', s=1)

        sample4x4x512 = SimpleSamplingLayer()([mean_4x4x512, stddev_4x4x512, mask4x4x512])
        # net = concatenate([net, sample4x4x512])
        net = sample4x4x512

        # Stage 7
        net = decoder_convolutional_block(net, f=3, filters=[512, 512, 256], stage=7, block='a', s=2)
        # net = decoder_convolutional_block(net, f=3, filters=[512, 512, 256], stage=7, block='b', s=1)

        sample8x8x256 = SimpleSamplingLayer()([mean_8x8x256, stddev_8x8x256, mask8x8x256])
        net = concatenate([net, sample8x8x256])

        # Stage 8
        net = decoder_convolutional_block(net, f=3, filters=[256, 256, 128], stage=8, block='a', s=2)
        # net = decoder_convolutional_block(net, f=3, filters=[256, 256, 128], stage=8, block='b', s=1)

        sample16x16x128 = SimpleSamplingLayer()([mean_16x16x128, stddev_16x16x128, mask16x16x128])
        net = concatenate([net, sample16x16x128])

        # Stage 9
        net = decoder_convolutional_block(net, f=3, filters=[128, 128, 64], stage=9, block='a', s=2)
        # net = decoder_convolutional_block(net, f=3, filters=[128, 128, 64], stage=9, block='b', s=1)

        sample32x32x64 = SimpleSamplingLayer()([mean_32x32x64, stddev_32x32x64, mask32x32x64])
        net = concatenate([net, sample32x32x64])

        # Stage 10
        net = decoder_convolutional_block(net, f=3, filters=[64, 64, 32], stage=10, block='a', s=2)
        # net = decoder_convolutional_block(net, f=3, filters=[64, 64, 32], stage=10, block='b', s=1)
        net = decoder_convolutional_block(net, f=3, filters=[32, 32, 16], stage=10, block='c', s=1)
        net = decoder_convolutional_block(net, f=3, filters=[16, 16, 8], stage=10, block='c', s=1)

        net = Conv2D(filters=3, kernel_size=3, use_bias=False, name="final_convolution",
                     data_format='channels_last', padding='same')(net)

        return [mean_4x4x512, stddev_4x4x512,
                mean_8x8x256, stddev_8x8x256,
                mean_16x16x128, stddev_16x16x128,
                mean_32x32x64, stddev_32x32x64,
                mask4x4x512,
                mask8x8x256,
                mask16x16x128,
                mask32x32x64], net

    def Build(self):
        decoder_input, decoder_output = self.layers()
        return Model(inputs=decoder_input, outputs=decoder_output)


class NVAEAutoEncoder64(Architecture):
    def __init__(self, hps,
                 batch_size=16,
                 encoder_frames_no=30):
        self.hps = hps
        self.encoder_frames_no = encoder_frames_no

        k.clear_session()

        self.encoder = NVAEEncoder64(hps,
                                     batch_size=batch_size,
                                     frames_no=encoder_frames_no)
        self.encoder_model = self.encoder.model

        self.mask_encoder = MaskEncoder64(hps,
                                          batch_size=batch_size)
        self.mask_encoder_model = self.mask_encoder.model

        self.decoder = NVAEDecoder64(hps,
                                     batch_size=batch_size)
        self.decoder_model = self.decoder.model

        super(NVAEAutoEncoder64, self).__init__(input_shape=(64, 64, 3),
                                                batch_size=batch_size,
                                                frames_no=encoder_frames_no)
        # Loss Function
        self.kl_weight = k.variable(self.hps['kl_weight_start'], name='kl_weight', dtype=np.float32)
        self.mask_kl_weight = k.variable(self.hps['mask_kl_weight_start'], name='mask_kl_weight', dtype=np.float32)
        self.face_metric = FaceMetric(None, gamma=self.hps['gamma']).get_loss_from_batch
        self.loss_func = self.model_loss()

    def Build(self):
        sequence_input = Input((self.encoder_frames_no, 64, 64, 3), self.batch_size,
                               name="nvae_seq_input")
        mask_input = Input((64, 64, 1), self.batch_size,
                           name="nvae_mask_input")
        encoder_output = self.encoder_model(sequence_input)

        self.mean_4x4x512 = encoder_output[0]
        self.stddev_4x4x512 = encoder_output[1]
        self.mean_8x8x256 = encoder_output[2]
        self.stddev_8x8x256 = encoder_output[3]
        self.mean_16x16x128 = encoder_output[4]
        self.stddev_16x16x128 = encoder_output[5]
        self.mean_32x32x64 = encoder_output[6]
        self.stddev_32x32x64 = encoder_output[7]

        mask_encoder_output = self.mask_encoder_model(mask_input)
        self.mask4x4x512 = mask_encoder_output[0]
        self.mask8x8x256 = mask_encoder_output[1]
        self.mask16x16x128 = mask_encoder_output[2]
        self.mask32x32x64 = mask_encoder_output[3]

        decoder_output = self.decoder_model(encoder_output + mask_encoder_output)
        net = DummyMaskLayer()(decoder_output)
        return Model(inputs=[sequence_input, mask_input], outputs=net)

    @staticmethod
    def calculate_kl_loss(mu, sigma):
        """ Function to calculate the KL loss term.
         Considers the tolerance value for which optimization for KL should stop """
        # kullback Leibler loss between normal distributions
        kl_cost = -0.5 * k.mean(1.0 + sigma - k.square(mu) - k.exp(sigma))
        return kl_cost

    @staticmethod
    def calculate_mse(tensor1, tensor2):
        return k.mean(k.square(tensor1 - tensor2))

    def mask_mse_loss(self, *args, **kwargs):
        return self.calculate_mse(k.zeros(self.mask4x4x512.shape), self.mask4x4x512) + \
               self.calculate_mse(k.zeros(self.mask8x8x256.shape), self.mask8x8x256) + \
               self.calculate_mse(k.zeros(self.mask16x16x128.shape), self.mask16x16x128) + \
               self.calculate_mse(k.zeros(self.mask32x32x64.shape), self.mask32x32x64)

    def face_kl_loss(self, *args, **kwargs):
        return self.calculate_kl_loss(self.mean_4x4x512, self.stddev_4x4x512) + \
               self.calculate_kl_loss(self.mean_8x8x256, self.stddev_8x8x256) + \
               self.calculate_kl_loss(self.mean_16x16x128, self.stddev_16x16x128) + \
               self.calculate_kl_loss(self.mean_32x32x64, self.stddev_32x32x64)

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # KL loss
        face_encoder_kl_loss = self.face_kl_loss
        mask_encoder_mse_loss = self.mask_mse_loss
        # Reconstruction loss
        md_loss_func = self.face_metric

        # KL weight (to be used by total loss and by annealing scheduler)
        kl_weight = self.kl_weight
        mask_kl_weight = self.mask_kl_weight

        def _model_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            md_loss = md_loss_func(y_true, y_pred)
            # Full loss
            result_loss = mask_kl_weight * mask_encoder_mse_loss() + kl_weight * face_encoder_kl_loss() + md_loss
            return result_loss

        return _model_loss

    def summary(self):
        print("Encoder summary:")
        self.encoder_model.summary()
        print("Decoder summary:")
        self.decoder_model.summary()
        print("Model summary:")
        self.model.summary()


def test_summary():
    hps = {'kl_weight_start': 1e-4,  # KL start weight when annealing.
           'kl_decay_rate': 0.9995,  # KL annealing decay rate per minibatch.
           'mask_kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
           'mask_kl_weight_start': 0.001,  # KL start weight when annealing.
           'mask_kl_decay_rate': 0.995,  # KL annealing decay rate per minibatch.
           "gamma": 1.0}
    auto_encoder = NVAEAutoEncoder64(hps,
                                     batch_size=4,
                                     encoder_frames_no=30)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
