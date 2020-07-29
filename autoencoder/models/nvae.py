import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import BatchNormalization, ConvLSTM2D, Dropout, \
    Flatten, MaxPool2D, MaxPooling2D, Reshape, TimeDistributed

from autoencoder.metric.metric import FaceMetric
from autoencoder.models.spectral_norm import ConvSN2D, ConvSN2DTranspose, DenseSN
from autoencoder.models.utils import DummyMaskLayer, EncoderResidualLayer, MaskResidualLayer, NVAEResidualLayer, \
    SmallEncoderResidualLayer, \
    SwishLayer, RelativeMeanLayer, RelativeStddevLayer, SimpleSamplingLayer


class Architecture(object):
    """
    generic architecture template
    """

    def __init__(self, input_shape=None, batch_size=None, latent_size=None, frames_no=None):
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
        self.latent_size = latent_size
        self.frames_no = frames_no
        self.real_input_shape = (self.frames_no,) + self.input_shape
        self.dropout = 0.5
        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class NVAEEncoder128(Architecture):
    def __init__(self, hps, batch_size=16, frames_no=30):
        self.hps = hps

        self.skip_128x128x16 = None
        self.skip_64x64x32 = None
        self.skip_32x32x64 = None
        self.skip_16x16x128 = None
        self.skip_8x8x256 = None
        self.skip_4x4x512 = None

        self.mean_1024 = None
        self.stddev_1024 = None

        super(NVAEEncoder128, self).__init__(input_shape=(128, 128, 3),
                                             batch_size=batch_size,
                                             frames_no=frames_no,
                                             latent_size=1024)

    def layers(self):
        input_layer = Input(self.real_input_shape, self.batch_size)

        # 128x128x3
        net = TimeDistributed(ConvSN2D(filters=16, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(input_layer)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_128x128x16 = EncoderResidualLayer(depth=16, name="skip_128x128x16")(net)

        net = TimeDistributed(ConvSN2D(filters=16, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 64x64x32
        net = TimeDistributed(ConvSN2D(filters=32, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_64x64x32 = EncoderResidualLayer(depth=32, name="skip_64x64x32")(net)

        net = TimeDistributed(ConvSN2D(filters=32, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 32x32x64
        net = TimeDistributed(ConvSN2D(filters=64, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_32x32x64 = EncoderResidualLayer(depth=64, name="skip_32x32x64")(net)

        net = TimeDistributed(ConvSN2D(filters=64, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 16x16x64
        net = TimeDistributed(ConvSN2D(filters=128, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_16x16x128 = EncoderResidualLayer(depth=128, name="skip_16x16x128")(net)

        net = TimeDistributed(ConvSN2D(filters=128, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 8x8x128
        net = TimeDistributed(ConvSN2D(filters=256, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_8x8x256 = EncoderResidualLayer(depth=256, name="skip_8x8x256")(net)

        net = TimeDistributed(ConvSN2D(filters=256, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 4x4x256
        net = TimeDistributed(ConvSN2D(filters=512, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        # skip connection
        self.skip_4x4x512 = EncoderResidualLayer(depth=512, name="skip_4x4x512")(net)

        net = TimeDistributed(ConvSN2D(filters=512, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        mean = EncoderResidualLayer(depth=self.latent_size, name="mean_2x2x1024")(net)
        mean = MaxPool2D(pool_size=(2, 2),
                         name="mean_max_pooling")(mean)
        mean = Flatten(name="mean_flatten")(mean)
        # mean = DenseSN(self.latent_size,
        #                name="mean")(mean)
        self.mean_1024 = mean
        # stddev = ConvSN2D(filters=self.latent_size, kernel_size=(1, 1),
        #                   padding='same', name="stddev_convolution")(net)

        stddev = EncoderResidualLayer(depth=self.latent_size, name="stddev_2x2x1024")(net)
        stddev = MaxPool2D(pool_size=(2, 2),
                           name="stddev_max_pooling")(stddev)
        stddev = Flatten(name="stddev_flatten")(stddev)
        # stddev = DenseSN(self.latent_size,
        #                  name="stddev")(stddev)
        self.stddev_1024 = stddev

        return input_layer, [self.mean_1024, self.stddev_1024,
                             self.skip_4x4x512,
                             self.skip_8x8x256,
                             self.skip_16x16x128,
                             self.skip_32x32x64,
                             self.skip_64x64x32,
                             self.skip_128x128x16]

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class MaskEncoder128(Architecture):
    def __init__(self, hps, batch_size=16, frames_no=30):
        self.hps = hps
        self.mask128x128x16 = None
        self.mask64x64x32 = None
        self.mask32x32x64 = None
        self.mask16x16x128 = None
        self.mask8x8x256 = None
        self.mask4x4x512 = None
        self.mask_1024 = None
        super(MaskEncoder128, self).__init__(input_shape=(128, 128, 1),
                                             batch_size=batch_size,
                                             frames_no=frames_no,
                                             latent_size=1024)

    def layers(self):
        mask_input = Input(self.input_shape, self.batch_size, name="mask_input")

        ########################
        # Mask encoder network #
        ########################

        # 128x128x3
        mask_net = ConvSN2D(filters=16, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_input)
        self.mask128x128x16 = MaskResidualLayer(depth=16)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 64x64x16
        mask_net = ConvSN2D(filters=32, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        self.mask64x64x32 = MaskResidualLayer(depth=32)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 32x32x32
        mask_net = ConvSN2D(filters=64, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        self.mask32x32x64 = MaskResidualLayer(depth=64)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 16x16x64
        mask_net = ConvSN2D(filters=128, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        self.mask16x16x128 = MaskResidualLayer(depth=128)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 8x8x128
        mask_net = ConvSN2D(filters=256, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        self.mask8x8x256 = MaskResidualLayer(depth=256)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 4x4x256
        mask_net = ConvSN2D(filters=512, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        self.mask4x4x512 = MaskResidualLayer(depth=512)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 2x2x512
        mask_net = ConvSN2D(filters=1024, kernel_size=(3, 3),
                            data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        # 1x1x1024
        mask_net = Dropout(self.dropout)(mask_net)
        mask_net = Flatten(name="mask_flatten")(mask_net)
        mask_net = DenseSN(self.latent_size, name="epsilon_DenseSN")(mask_net)
        self.mask_1024 = mask_net

        return mask_input, [self.mask_1024,
                            self.mask4x4x512,
                            self.mask8x8x256,
                            self.mask16x16x128,
                            self.mask32x32x64,
                            self.mask64x64x32,
                            self.mask128x128x16]

    def Build(self):
        encoder_input, encoder_output = self.layers()
        return Model(inputs=encoder_input, outputs=encoder_output)


class NVAEDecoder128(Architecture):
    def __init__(self, hps, batch_size=4):
        self.hps = hps

        self.mean_128x128x16 = None
        self.stddev_128x128x16 = None

        self.mean_64x64x32 = None
        self.stddev_64x64x32 = None

        self.mean_32x32x64 = None
        self.stddev_32x32x64 = None

        self.mean_16x16x128 = None
        self.stddev_16x16x128 = None

        self.mean_8x8x256 = None
        self.stddev_8x8x256 = None

        self.mean_4x4x512 = None
        self.stddev_4x4x512 = None

        self.mean_1024 = None
        self.stddev_1024 = None

        super(NVAEDecoder128, self).__init__(input_shape=(128, 128, 3),
                                             batch_size=batch_size,
                                             latent_size=1024)

    def layers(self):
        # Face encoder inputs
        self.mean_1024 = Input(self.latent_size, self.batch_size, name="mean_1024")
        self.stddev_1024 = Input(self.latent_size, self.batch_size, name="stddev_1024")
        skip_4x4x512 = Input((4, 4, 512), self.batch_size, name="skip_4x4x512")
        skip_8x8x256 = Input((8, 8, 256), self.batch_size, name="skip_8x8x256")
        skip_16x16x128 = Input((16, 16, 128), self.batch_size, name="skip_16x16x128")
        skip_32x32x64 = Input((32, 32, 64), self.batch_size, name="skip_32x32x64")
        skip_64x64x32 = Input((64, 64, 32), self.batch_size, name="skip_64x64x32")
        skip_128x128x16 = Input((128, 128, 16), self.batch_size, name="skip_128x128x16")

        # Mask encoder inputs
        mask_1024 = Input(self.latent_size, self.batch_size, name="mask_1024")
        mask4x4x512 = Input((4, 4, 512), self.batch_size, name="mask4x4x512")
        mask8x8x256 = Input((8, 8, 256), self.batch_size, name="mask8x8x256")
        mask16x16x128 = Input((16, 16, 128), self.batch_size, name="mask16x16x128")
        mask32x32x64 = Input((32, 32, 64), self.batch_size, name="mask32x32x64")
        mask64x64x32 = Input((64, 64, 32), self.batch_size, name="mask64x64x32")
        mask128x128x16 = Input((128, 128, 16), self.batch_size, name="mask128x128x16")

        sample = SimpleSamplingLayer()([self.mean_1024, self.stddev_1024, mask_1024])

        ###################
        # Decoder network #
        ###################
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latent_size))(sample)
        net = SwishLayer()(net)

        # 1x1x1024
        net = ConvSN2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(net)
        # 2x2x1024
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=1024, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # decode distribution
        decoded_4x4x512 = ConvSN2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_4x4x1024 = concatenate([skip_4x4x512, decoded_4x4x512])
        mean_4x4x512 = NVAEResidualLayer(512)(decoded_4x4x1024)
        stddev_4x4x512 = NVAEResidualLayer(512)(decoded_4x4x1024)
        # count delta
        upscaled_mean_1024 = Reshape((1, 1, 1024))(self.mean_1024)
        upscaled_mean_1024 = ConvSN2DTranspose(512, (3, 3), strides=(4, 4), padding='same')(upscaled_mean_1024)
        self.mean_4x4x512 = RelativeMeanLayer()([mean_4x4x512,
                                                 upscaled_mean_1024])
        upscaled_stddev_1024 = Reshape((1, 1, 1024))(self.stddev_1024)
        upscaled_stddev_1024 = ConvSN2DTranspose(512, (3, 3), strides=(4, 4), padding='same')(upscaled_stddev_1024)
        self.stddev_4x4x512 = RelativeStddevLayer()([stddev_4x4x512,
                                                     upscaled_stddev_1024])
        # sample distribution
        sample_4x4x512 = SimpleSamplingLayer()([self.mean_4x4x512,
                                                self.stddev_4x4x512,
                                                mask4x4x512])

        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=1024, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 2x2x1024
        net = ConvSN2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(net)
        # 4x4x512

        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_4x4x512])

        # 4x4x512
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=512, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # decode distribution
        decoded_8x8x256 = ConvSN2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_8x8x256 = concatenate([skip_8x8x256, decoded_8x8x256])
        mean_8x8x256 = NVAEResidualLayer(256)(decoded_8x8x256)
        stddev_8x8x256 = NVAEResidualLayer(256)(decoded_8x8x256)
        # count delta
        upscaled_mean_4x4x512 = ConvSN2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(self.mean_4x4x512)
        self.mean_8x8x256 = RelativeMeanLayer()([mean_8x8x256,
                                                 upscaled_mean_4x4x512])
        upscaled_stddev_4x4x512 = ConvSN2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(self.stddev_4x4x512)
        self.stddev_8x8x256 = RelativeStddevLayer()([stddev_8x8x256,
                                                     upscaled_stddev_4x4x512])
        # sample distribution
        sample_8x8x256 = SimpleSamplingLayer()([self.mean_8x8x256,
                                                self.stddev_8x8x256,
                                                mask8x8x256])

        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=512, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 4x4x512
        net = ConvSN2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(net)
        # 8x8x256
        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_8x8x256])
        # 8x8x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=256, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # decode distribution
        decoded_16x16x128 = ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_16x16x128 = concatenate([skip_16x16x128, decoded_16x16x128])
        mean_16x16x128 = NVAEResidualLayer(128)(decoded_16x16x128)
        stddev_16x16x128 = NVAEResidualLayer(128)(decoded_16x16x128)
        # count delta
        upscaled_mean_8x8x256 = ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(self.mean_8x8x256)
        self.mean_16x16x128 = RelativeMeanLayer()([mean_16x16x128,
                                                   upscaled_mean_8x8x256])
        upscaled_stddev_8x8x256 = ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(self.stddev_8x8x256)
        self.stddev_16x16x128 = RelativeStddevLayer()([stddev_16x16x128,
                                                       upscaled_stddev_8x8x256])
        # sample distribution
        sample_16x16x128 = SimpleSamplingLayer()([self.mean_16x16x128,
                                                  self.stddev_16x16x128,
                                                  mask16x16x128])

        # 8x8x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=256, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 8x8x256
        net = ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(net)
        # 16x16x128
        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_16x16x128])
        # 16x16x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=128, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 16x16x128

        # decode distribution
        decoded_32x32x64 = ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_32x32x64 = concatenate([skip_32x32x64, decoded_32x32x64])
        mean_32x32x64 = NVAEResidualLayer(64)(decoded_32x32x64)
        stddev_32x32x64 = NVAEResidualLayer(64)(decoded_32x32x64)
        # count delta
        upscaled_mean_16x16x128 = ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(self.mean_16x16x128)
        self.mean_32x32x64 = RelativeMeanLayer()([mean_32x32x64,
                                                  upscaled_mean_16x16x128])
        upscaled_stddev_16x16x128 = ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(self.stddev_16x16x128)
        self.stddev_32x32x64 = RelativeStddevLayer()([stddev_32x32x64,
                                                      upscaled_stddev_16x16x128])
        # sample distribution
        sample_32x32x64 = SimpleSamplingLayer()([self.mean_32x32x64,
                                                 self.stddev_32x32x64,
                                                 mask32x32x64])

        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=128, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 16x16x128
        net = ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(net)
        # 32x32x64

        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_32x32x64])

        # 32x32x128
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=64, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 32x32x64

        # decode distribution
        decoded_64x64x32 = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_64x64x32 = concatenate([skip_64x64x32, decoded_64x64x32])
        mean_64x64x32 = NVAEResidualLayer(32)(decoded_64x64x32)
        stddev_64x64x32 = NVAEResidualLayer(32)(decoded_64x64x32)
        # count delta
        upscaled_mean_32x32x64 = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(self.mean_32x32x64)
        self.mean_64x64x32 = RelativeMeanLayer()([mean_64x64x32,
                                                  upscaled_mean_32x32x64])
        upscaled_stddev_32x32x64 = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(self.stddev_32x32x64)
        self.stddev_64x64x32 = RelativeStddevLayer()([stddev_64x64x32,
                                                      upscaled_stddev_32x32x64])
        # sample distribution
        sample_64x64x32 = SimpleSamplingLayer()([self.mean_64x64x32,
                                                 self.stddev_64x64x32,
                                                 mask64x64x32])

        net = Dropout(self.dropout)(net)
        # 32x32x128
        net = ConvSN2D(filters=64, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 32x32x64
        net = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net)
        # 64x64x32
        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_64x64x32])
        # 64x64x64
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=32, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 64x64x32

        # decode distribution
        decoded_128x128x16 = ConvSN2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)
        decoded_128x128x16 = concatenate([skip_128x128x16, decoded_128x128x16])
        mean_128x128x16 = NVAEResidualLayer(16)(decoded_128x128x16)
        stddev_128x128x16 = NVAEResidualLayer(16)(decoded_128x128x16)
        # count delta
        upscaled_mean_64x64x32 = ConvSN2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(self.mean_64x64x32)
        self.mean_128x128x16 = RelativeMeanLayer()([mean_128x128x16,
                                                    upscaled_mean_64x64x32])
        upscaled_stddev_64x64x32 = ConvSN2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(self.stddev_64x64x32)
        self.stddev_128x128x16 = RelativeStddevLayer()([stddev_128x128x16,
                                                        upscaled_stddev_64x64x32])
        # sample distribution
        sample_128x128x16 = SimpleSamplingLayer()([self.mean_128x128x16,
                                                   self.stddev_128x128x16,
                                                   mask128x128x16])

        net = Dropout(self.dropout)(net)
        # 64x64x64
        net = ConvSN2D(filters=32, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 64x64x32
        net = ConvSN2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)
        # concatenate sample from distribution with normal layer
        net = concatenate([net, sample_128x128x16])
        # 128x128x32
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=16, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        net = ConvSN2D(filters=3, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        return [self.mean_1024, self.stddev_1024,
                # face encodings
                skip_4x4x512,
                skip_8x8x256,
                skip_16x16x128,
                skip_32x32x64,
                skip_64x64x32,
                skip_128x128x16,
                # mask
                mask_1024,
                mask4x4x512,
                mask8x8x256,
                mask16x16x128,
                mask32x32x64,
                mask64x64x32,
                mask128x128x16], [net,
                                  self.mean_1024, self.stddev_1024,
                                  self.mean_4x4x512, self.stddev_4x4x512,
                                  self.mean_8x8x256, self.stddev_8x8x256,
                                  self.mean_16x16x128, self.stddev_16x16x128,
                                  self.mean_32x32x64, self.stddev_32x32x64,
                                  self.mean_64x64x32, self.stddev_64x64x32,
                                  self.mean_128x128x16, self.stddev_128x128x16]

    def Build(self):
        decoder_input, decoder_output = self.layers()
        return Model(inputs=decoder_input, outputs=decoder_output)


class NVAEAutoEncoder128(Architecture):
    def __init__(self, hps,
                 batch_size=16,
                 encoder_frames_no=30):
        self.hps = hps
        self.encoder_frames_no = encoder_frames_no

        K.clear_session()

        self.encoder = NVAEEncoder128(hps,
                                      batch_size=batch_size,
                                      frames_no=encoder_frames_no)
        self.encoder_model = self.encoder.model

        self.mask_encoder = MaskEncoder128(hps,
                                           batch_size=batch_size)
        self.mask_encoder_model = self.mask_encoder.model

        self.decoder = NVAEDecoder128(hps,
                                      batch_size=batch_size)
        self.decoder_model = self.decoder.model

        super(NVAEAutoEncoder128, self).__init__(input_shape=(128, 128, 3),
                                                 batch_size=batch_size,
                                                 latent_size=1024,
                                                 frames_no=encoder_frames_no)
        # Loss Function
        self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight', dtype=np.float32)
        self.mask_kl_weight = K.variable(self.hps['mask_kl_weight_start'], name='mask_kl_weight', dtype=np.float32)
        self.face_metric = FaceMetric(None, gamma=self.hps['gamma']).get_loss_from_batch
        self.loss_func = self.model_loss()

    def Build(self):
        sequence_input = Input((self.encoder_frames_no, 128, 128, 3), self.batch_size,
                               name="nvae_seq_input")
        mask_input = Input((128, 128, 1), self.batch_size,
                           name="nvae_mask_input")
        encoder_output = self.encoder_model(sequence_input)
        self.mean_1024 = encoder_output[0]
        self.stddev_1024 = encoder_output[1]
        self.skip_4x4x512 = encoder_output[2]
        self.skip_8x8x256 = encoder_output[3]
        self.skip_16x16x128 = encoder_output[4]
        self.skip_32x32x64 = encoder_output[5]
        self.skip_64x64x32 = encoder_output[6]
        self.skip_128x128x16 = encoder_output[7]

        mask_encoder_output = self.mask_encoder_model(mask_input)
        self.mask_1024 = mask_encoder_output[0]
        self.mask4x4x512 = mask_encoder_output[1]
        self.mask8x8x256 = mask_encoder_output[2]
        self.mask16x16x128 = mask_encoder_output[3]
        self.mask32x32x64 = mask_encoder_output[4]
        self.mask64x64x32 = mask_encoder_output[5]
        self.mask128x128x16 = mask_encoder_output[6]

        decoder_output = self.decoder_model(encoder_output + mask_encoder_output)

        net = decoder_output[0]
        self.mean_1024 = decoder_output[1]
        self.stddev_1024 = decoder_output[2]
        self.mean_4x4x512 = decoder_output[3]
        self.stddev_4x4x512 = decoder_output[4]
        self.mean_8x8x256 = decoder_output[5]
        self.stddev_8x8x256 = decoder_output[6]
        self.mean_16x16x128 = decoder_output[7]
        self.stddev_16x16x128 = decoder_output[8]
        self.mean_32x32x64 = decoder_output[9]
        self.stddev_32x32x64 = decoder_output[10]
        self.mean_64x64x32 = decoder_output[11]
        self.stddev_64x64x32 = decoder_output[12]
        self.mean_128x128x16 = decoder_output[13]
        self.stddev_128x128x16 = decoder_output[14]

        net = DummyMaskLayer()(net)
        return Model(inputs=[sequence_input, mask_input], outputs=net)

    def calculate_kl_loss(self, mu, sigma):
        """ Function to calculate the KL loss term.
         Considers the tolerance value for which optimization for KL should stop """
        # kullback Leibler loss between normal distributions
        kl_cost = -0.5 * K.mean(1.0 + sigma - K.square(mu) - K.exp(sigma))
        # return K.maximum(kl_cost, self.hps['kl_tolerance'])
        return kl_cost

    def mask_kl_loss(self, *args, **kwargs):
        return self.calculate_kl_loss(K.zeros(self.mask_1024.shape), self.mask_1024) + \
               self.calculate_kl_loss(K.zeros(self.mask4x4x512.shape), self.mask4x4x512) + \
               self.calculate_kl_loss(K.zeros(self.mask8x8x256.shape), self.mask8x8x256) + \
               self.calculate_kl_loss(K.zeros(self.mask16x16x128.shape), self.mask16x16x128) + \
               self.calculate_kl_loss(K.zeros(self.mask32x32x64.shape), self.mask32x32x64) + \
               self.calculate_kl_loss(K.zeros(self.mask64x64x32.shape), self.mask64x64x32) + \
               self.calculate_kl_loss(K.zeros(self.mask128x128x16.shape), self.mask128x128x16)

    def face_kl_loss(self, *args, **kwargs):
        return self.calculate_kl_loss(self.mean_1024, self.stddev_1024) + \
               self.calculate_kl_loss(self.mean_4x4x512, self.stddev_4x4x512) + \
               self.calculate_kl_loss(self.mean_8x8x256, self.stddev_8x8x256) + \
               self.calculate_kl_loss(self.mean_16x16x128, self.stddev_16x16x128) + \
               self.calculate_kl_loss(self.mean_32x32x64, self.stddev_32x32x64) + \
               self.calculate_kl_loss(self.mean_64x64x32, self.stddev_64x64x32) + \
               self.calculate_kl_loss(self.mean_128x128x16, self.stddev_128x128x16)

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # KL loss
        face_encoder_kl_loss = self.face_kl_loss
        mask_encoder_kl_loss = self.mask_kl_loss
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
            result_loss = mask_kl_weight * mask_encoder_kl_loss() + kl_weight * face_encoder_kl_loss() + md_loss
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
    auto_encoder = NVAEAutoEncoder128({},
                                      batch_size=4,
                                      encoder_frames_no=30)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
