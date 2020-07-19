from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import BatchNormalization, ConvLSTM2D, Dropout, \
    Flatten, MaxPool2D, MaxPooling2D, Reshape, TimeDistributed

from autoencoder.models.spectral_norm import ConvSN2D, DenseSN, ConvSN2DTranspose
from autoencoder.models.utils import DummyMaskLayer, EncoderResidualLayer, EpsilonLayer, NVAEResidualLayer, \
    SampleLayer, SwishLayer, MaskResidualLayer


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
        self.dropout = 0.1

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class NVAEEncoder128(Architecture):
    def __init__(self, batch_size=16, frames_no=30):
        super().__init__(input_shape=(128, 128, 3),
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
        net = TimeDistributed(ConvSN2D(filters=16, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_64x64x32 = BatchNormalization()(net)
        skip_64x64x32 = TimeDistributed(SwishLayer())(skip_64x64x32)
        skip_64x64x32 = ConvLSTM2D(filters=32, kernel_size=(3, 3), data_format='channels_last',
                                   padding='same', return_sequences=False)(skip_64x64x32)
        mean_64x64x32 = NVAEResidualLayer(depth=32, name="mean_64x64x32")(skip_64x64x32)
        stddev_64x64x32 = NVAEResidualLayer(depth=32, name="stddev_64x64x32")(skip_64x64x32)

        # 64x64x32
        net = TimeDistributed(ConvSN2D(filters=32, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(ConvSN2D(filters=32, kernel_size=3,
                                       use_bias=False, data_format='channels_last',
                                       padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_32x32x64 = BatchNormalization()(net)
        skip_32x32x64 = TimeDistributed(SwishLayer())(skip_32x32x64)
        skip_32x32x64 = ConvLSTM2D(filters=64, kernel_size=(3, 3), data_format='channels_last',
                                   padding='same', return_sequences=False)(skip_32x32x64)
        mean_32x32x64 = NVAEResidualLayer(depth=64, name="mean_32x32x64")(skip_32x32x64)
        stddev_32x32x64 = NVAEResidualLayer(depth=64, name="stddev_32x32x64")(skip_32x32x64)

        # 32x32x64
        net = TimeDistributed(ConvSN2D(filters=64, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        net = TimeDistributed(ConvSN2D(filters=64, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_16x16x128 = BatchNormalization()(net)
        skip_16x16x128 = TimeDistributed(SwishLayer())(skip_16x16x128)
        skip_16x16x128 = ConvLSTM2D(filters=128, kernel_size=(3, 3), data_format='channels_last',
                                    padding='same', return_sequences=False)(skip_16x16x128)
        mean_16x16x128 = NVAEResidualLayer(depth=128, name="mean_16x16x128")(skip_16x16x128)
        stddev_16x16x128 = NVAEResidualLayer(depth=128, name="stddev_16x16x128")(skip_16x16x128)

        # 16x16x64
        net = TimeDistributed(ConvSN2D(filters=128, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = ConvLSTM2D(filters=128, kernel_size=(3, 3), data_format='channels_last',
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_8x8x256 = BatchNormalization()(net)
        skip_8x8x256 = TimeDistributed(SwishLayer())(skip_8x8x256)
        skip_8x8x256 = ConvLSTM2D(filters=256, kernel_size=(3, 3), data_format='channels_last',
                                  padding='same', return_sequences=False)(skip_8x8x256)
        mean_8x8x256 = NVAEResidualLayer(depth=256, name="mean_8x8x256")(skip_8x8x256)
        stddev_8x8x256 = NVAEResidualLayer(depth=256, name="stddev_8x8x256")(skip_8x8x256)

        # 8x8x128
        net = TimeDistributed(ConvSN2D(filters=256, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = ConvLSTM2D(filters=256, kernel_size=(3, 3), data_format='channels_last',
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_4x4x512 = BatchNormalization()(net)
        skip_4x4x512 = TimeDistributed(SwishLayer())(skip_4x4x512)
        skip_4x4x512 = ConvLSTM2D(filters=512, kernel_size=(3, 3), data_format='channels_last',
                                  padding='same', return_sequences=False)(skip_4x4x512)
        mean_4x4x512 = NVAEResidualLayer(depth=512, name="mean_4x4x512")(skip_4x4x512)
        stddev_4x4x512 = NVAEResidualLayer(depth=512, name="stddev_4x4x512")(skip_4x4x512)

        # 4x4x256
        net = TimeDistributed(ConvSN2D(filters=512, kernel_size=(3, 3),
                                       data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        net = ConvLSTM2D(filters=512, kernel_size=(3, 3), data_format='channels_last',
                         padding='same', return_sequences=False)(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 2x2x512

        # variational encoder output (distributions)
        mean = ConvSN2D(filters=self.latent_size, kernel_size=(1, 1),
                        padding='same', name="mean_convolution")(net)
        mean = MaxPool2D(pool_size=(2, 2),
                         name="mean_max_pooling")(mean)
        mean = Flatten(name="mean_flatten")(mean)
        mean = DenseSN(self.latent_size,
                       name="mean")(mean)
        stddev = ConvSN2D(filters=self.latent_size, kernel_size=(1, 1),
                          padding='same', name="stddev_convolution")(net)
        stddev = MaxPool2D(pool_size=(2, 2),
                           name="stddev_max_pooling")(stddev)
        stddev = Flatten(name="stddev_flatten")(stddev)
        stddev = DenseSN(self.latent_size,
                         name="stddev")(stddev)

        return input_layer, [mean, stddev,
                             mean_4x4x512, stddev_4x4x512,
                             mean_8x8x256, stddev_8x8x256,
                             mean_16x16x128, stddev_16x16x128,
                             mean_32x32x64, stddev_32x32x64,
                             mean_64x64x32, stddev_64x64x32]

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class NVAEDecoder128(Architecture):
    def __init__(self,
                 alpha=1.0,
                 beta=1.0,
                 batch_size=4):
        self.alpha = alpha
        self.beta = beta
        self.mask_input_shape = (128, 128, 1)
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         latent_size=1024)

    def layers(self):
        mean_input = Input(self.latent_size, self.batch_size, name="mean_input")
        stddev_input = Input(self.latent_size, self.batch_size, name="stddev_input")
        mean_4x4x512 = Input((4, 4, 512), self.batch_size, name="mean_4x4x512")
        stddev_4x4x512 = Input((4, 4, 512), self.batch_size, name="stddev_4x4x512")
        mean_8x8x256 = Input((8, 8, 256), self.batch_size, name="mean_8x8x256")
        stddev_8x8x256 = Input((8, 8, 256), self.batch_size, name="stddev_8x8x256")
        mean_16x16x128 = Input((16, 16, 128), self.batch_size, name="mean_16x16x128")
        stddev_16x16x128 = Input((16, 16, 128), self.batch_size, name="stddev_16x16x128")
        mean_32x32x64 = Input((32, 32, 64), self.batch_size, name="mean_32x32x64")
        stddev_32x32x64 = Input((32, 32, 64), self.batch_size, name="stddev_32x32x64")
        mean_64x64x32 = Input((64, 64, 32), self.batch_size, name="mean_64x64x32")
        stddev_64x64x32 = Input((64, 64, 32), self.batch_size, name="stddev_64x64x32")

        mask_input = Input(self.mask_input_shape, self.batch_size, name="mask_input")

        ########################
        # Mask encoder network #
        ########################

        # 128x128x3
        mask_net = NVAEResidualLayer(depth=16)(mask_input)
        mask_net = NVAEResidualLayer(depth=16)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 64x64x16
        mask_net = NVAEResidualLayer(depth=32)(mask_net)
        mask_net = NVAEResidualLayer(depth=32)(mask_net)
        mask64x64x32 = MaskResidualLayer(depth=32)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 32x32x32
        mask_net = NVAEResidualLayer(depth=64)(mask_net)
        mask_net = NVAEResidualLayer(depth=64)(mask_net)
        mask32x32x64 = MaskResidualLayer(depth=64)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 16x16x64
        mask_net = NVAEResidualLayer(depth=128)(mask_net)
        mask_net = NVAEResidualLayer(depth=128)(mask_net)
        mask16x16x128 = MaskResidualLayer(depth=128)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 8x8x128
        mask_net = NVAEResidualLayer(depth=256)(mask_net)
        mask_net = NVAEResidualLayer(depth=256)(mask_net)
        mask8x8x256 = MaskResidualLayer(depth=256)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 4x4x256
        mask_net = NVAEResidualLayer(depth=512)(mask_net)
        mask_net = NVAEResidualLayer(depth=512)(mask_net)
        mask4x4x512 = MaskResidualLayer(depth=512)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        mask_net = Dropout(self.dropout)(mask_net)

        # 2x2x512
        mask_net = NVAEResidualLayer(depth=1024)(mask_net)
        mask_net = NVAEResidualLayer(depth=1024)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        # 1x1x1024
        mask_net = Dropout(self.dropout)(mask_net)
        mask_net = Flatten(name="mask_flatten")(mask_net)
        mask_net = DenseSN(self.latent_size, name="epsilon_DenseSN")(mask_net)

        epsilon = EpsilonLayer(alpha=self.alpha, name="epsilon_layer")(mask_net)

        previous_mean = MaxPooling2D(pool_size=4)(mean_4x4x512)
        previous_mean = NVAEResidualLayer(depth=1024)(previous_mean)
        previous_mean = Flatten(data_format='channels_last')(previous_mean)
        previous_stddev = MaxPooling2D(pool_size=4)(stddev_4x4x512)
        previous_stddev = NVAEResidualLayer(depth=1024)(previous_stddev)
        previous_stddev = Flatten(data_format='channels_last')(previous_stddev)
        sample = SampleLayer(beta=self.beta, relative=True,
                             name="sampling_layer")([mean_input, stddev_input, epsilon,
                                                     previous_mean, previous_stddev])

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
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=1024, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 2x2x1024
        net = ConvSN2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(net)
        # 4x4x512

        # sampling
        mean_4x4x512_flat = Flatten(data_format='channels_last')(mean_4x4x512)
        stddev_4x4x512_flat = Flatten(data_format='channels_last')(stddev_4x4x512)
        epsilon_4x4x512 = Flatten(data_format='channels_last')(mask4x4x512)
        epsilon_4x4x512 = EpsilonLayer(alpha=self.alpha, name="epsilon_4x4x512")(epsilon_4x4x512)
        previous_mean = MaxPooling2D(pool_size=2)(mean_8x8x256)
        previous_mean = NVAEResidualLayer(depth=512)(previous_mean)
        previous_mean = Flatten(data_format='channels_last')(previous_mean)
        previous_stddev = MaxPooling2D(pool_size=2)(stddev_8x8x256)
        previous_stddev = NVAEResidualLayer(depth=512)(previous_stddev)
        previous_stddev = Flatten(data_format='channels_last')(previous_stddev)
        sample_4x4x512 = SampleLayer(beta=self.beta, relative=True,
                                     name="sample_4x4x512")([mean_4x4x512_flat, stddev_4x4x512_flat, epsilon_4x4x512,
                                                             previous_mean, previous_stddev])
        sample_4x4x512 = Reshape((4, 4, 512))(sample_4x4x512)
        # 4x4x512
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=512, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 4x4x512
        net = concatenate([net, sample_4x4x512])
        # 4x4x1024
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=512, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 4x4x512
        net = ConvSN2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(net)
        # 8x8x256
        # sampling
        mean_8x8x256_flat = Flatten(data_format='channels_last')(mean_8x8x256)
        stddev_8x8x256_flat = Flatten(data_format='channels_last')(stddev_8x8x256)
        epsilon_8x8x256 = Flatten(data_format='channels_last')(mask8x8x256)
        epsilon_8x8x256 = EpsilonLayer(alpha=self.alpha, name="epsilon_8x8x256")(epsilon_8x8x256)
        previous_mean = MaxPooling2D(pool_size=2)(mean_16x16x128)
        previous_mean = NVAEResidualLayer(depth=256)(previous_mean)
        previous_mean = Flatten(data_format='channels_last')(previous_mean)
        previous_stddev = MaxPooling2D(pool_size=2)(stddev_16x16x128)
        previous_stddev = NVAEResidualLayer(depth=256)(previous_stddev)
        previous_stddev = Flatten(data_format='channels_last')(previous_stddev)
        sample_8x8x256 = SampleLayer(beta=self.beta, relative=True,
                                     name="sample_8x8x256")([mean_8x8x256_flat, stddev_8x8x256_flat, epsilon_8x8x256,
                                                             previous_mean, previous_stddev])
        sample_8x8x256 = Reshape((8, 8, 256))(sample_8x8x256)
        # 8x8x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=256, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 8x8x256
        net = concatenate([net, sample_8x8x256])
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=256, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 8x8x256
        net = ConvSN2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(net)
        # 16x16x128
        # sampling
        mean_16x16x128_flat = Flatten(data_format='channels_last')(mean_16x16x128)
        stddev_16x16x128_flat = Flatten(data_format='channels_last')(stddev_16x16x128)
        epsilon_16x16x128 = Flatten(data_format='channels_last')(mask16x16x128)
        epsilon_16x16x128 = EpsilonLayer(alpha=self.alpha, name="epsilon_16x16x128")(epsilon_16x16x128)
        previous_mean = MaxPooling2D(pool_size=2)(mean_32x32x64)
        previous_mean = NVAEResidualLayer(depth=128)(previous_mean)
        previous_mean = Flatten(data_format='channels_last')(previous_mean)
        previous_stddev = MaxPooling2D(pool_size=2)(stddev_32x32x64)
        previous_stddev = NVAEResidualLayer(depth=128)(previous_stddev)
        previous_stddev = Flatten(data_format='channels_last')(previous_stddev)
        sample_16x16x128 = SampleLayer(beta=self.beta, relative=True,
                                       name="sample_16x16x128")(
            [mean_16x16x128_flat, stddev_16x16x128_flat, epsilon_16x16x128,
             previous_mean, previous_stddev])
        sample_16x16x128 = Reshape((16, 16, 128))(sample_16x16x128)
        # 16x16x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=128, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 16x16x128
        net = concatenate([net, sample_16x16x128])
        # 16x16x256
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=128, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 16x16x128
        net = ConvSN2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(net)
        # 32x32x64
        # sampling
        mean_32x32x64_flat = Flatten(data_format='channels_last')(mean_32x32x64)
        stddev_32x32x64_flat = Flatten(data_format='channels_last')(stddev_32x32x64)
        epsilon_32x32x64 = Flatten(data_format='channels_last')(mask32x32x64)
        epsilon_32x32x64 = EpsilonLayer(alpha=self.alpha, name="epsilon_32x32x64")(epsilon_32x32x64)
        previous_mean = MaxPooling2D(pool_size=2)(mean_64x64x32)
        previous_mean = NVAEResidualLayer(depth=64)(previous_mean)
        previous_mean = Flatten(data_format='channels_last')(previous_mean)
        previous_stddev = MaxPooling2D(pool_size=2)(stddev_64x64x32)
        previous_stddev = NVAEResidualLayer(depth=64)(previous_stddev)
        previous_stddev = Flatten(data_format='channels_last')(previous_stddev)
        sample_32x32x64 = SampleLayer(beta=self.beta, relative=True,
                                      name="sample_32x32x64")(
            [mean_32x32x64_flat, stddev_32x32x64_flat, epsilon_32x32x64,
             previous_mean, previous_stddev])
        sample_32x32x64 = Reshape((32, 32, 64))(sample_32x32x64)
        # 32x32x128
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=64, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 32x32x64
        net = concatenate([net, sample_32x32x64])
        net = Dropout(self.dropout)(net)
        # 32x32x128
        net = ConvSN2D(filters=64, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 32x32x64
        net = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net)
        # 64x64x32
        # sampling
        mean_64x64x32_flat = Flatten(data_format='channels_last')(mean_64x64x32)
        stddev_64x64x32_flat = Flatten(data_format='channels_last')(stddev_64x64x32)
        epsilon_64x64x32 = Flatten(data_format='channels_last')(mask64x64x32)
        epsilon_64x64x32 = EpsilonLayer(alpha=self.alpha, name="epsilon_64x64x32")(epsilon_64x64x32)
        sample_64x64x32 = SampleLayer(beta=self.beta, name="sample_64x64x32")(
            [mean_64x64x32_flat, stddev_64x64x32_flat, epsilon_64x64x32])
        sample_64x64x32 = Reshape((64, 64, 32))(sample_64x64x32)
        # 64x64x64
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=32, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 64x64x32
        net = concatenate([net, sample_64x64x32])
        net = Dropout(self.dropout)(net)
        # 64x64x64
        net = ConvSN2D(filters=32, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 64x64x32
        net = ConvSN2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)
        # 128x128x16
        net = Dropout(self.dropout)(net)
        net = ConvSN2D(filters=16, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        net = ConvSN2D(filters=3, kernel_size=3, use_bias=False,
                       data_format='channels_last', padding='same')(net)
        return [mean_input, stddev_input,
                mean_4x4x512, stddev_4x4x512,
                mean_8x8x256, stddev_8x8x256,
                mean_16x16x128, stddev_16x16x128,
                mean_32x32x64, stddev_32x32x64,
                mean_64x64x32, stddev_64x64x32,
                mask_input], net

    def Build(self):
        decoder_input, decoder_output = self.layers()
        return Model(inputs=decoder_input, outputs=decoder_output)


class NVAEAutoEncoder128(object):
    def __init__(self, batch_size=16, alpha=0.2, beta=0.2,
                 encoder_frames_no=30):
        self.latent_size = 1024
        self.batch_size = batch_size
        self.encoder_frames_no = encoder_frames_no
        self.encoder = NVAEEncoder128(batch_size=batch_size,
                                      frames_no=encoder_frames_no)
        self.encoder_model = self.encoder.Build()
        self.decoder = NVAEDecoder128(batch_size=batch_size,
                                      alpha=alpha,
                                      beta=beta)
        self.decoder_model = self.decoder.Build()
        self.model = self.Build()

    def Build(self):
        sequence_input = Input((self.encoder_frames_no, 128, 128, 3), self.batch_size,
                               name="nvae_seq_input")
        mask_input = Input((128, 128, 1), self.batch_size,
                           name="nvae_mask_input")
        encoder_output = self.encoder_model(sequence_input)
        decoder_output = self.decoder_model(encoder_output + [mask_input])
        net = DummyMaskLayer()(decoder_output)
        return Model(inputs=[sequence_input, mask_input], outputs=net)

    def summary(self):
        print("Encoder summary:")
        self.encoder_model.summary()
        print("Decoder summary:")
        self.decoder_model.summary()
        print("Model summary:")
        self.model.summary()


def test_summary():
    auto_encoder = NVAEAutoEncoder128(batch_size=4,
                                      encoder_frames_no=30,
                                      alpha=0.1,
                                      beta=0.1)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
