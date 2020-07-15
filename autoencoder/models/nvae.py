from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, ConvLSTM2D, Dense, Dropout, \
    Flatten, MaxPool2D, MaxPooling2D, Reshape, TimeDistributed

from autoencoder.models.utils import DummyMaskLayer, EncoderResidualLayer, EpsilonLayer, NVAEResidualLayer, SampleLayer, \
    SwishLayer


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
        net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                     use_bias=False, data_format='channels_last',
                                     padding='same'))(input_layer)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                     use_bias=False, data_format='channels_last',
                                     padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_64x64x32 = EncoderResidualLayer(depth=32, name="skip_64x64x32")(net)

        # 64x64x32
        net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                     use_bias=False, data_format='channels_last',
                                     padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                     use_bias=False, data_format='channels_last',
                                     padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_32x32x64 = EncoderResidualLayer(depth=64, name="skip_32x32x64")(net)

        # 32x32x64
        net = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3),
                                     data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)

        net = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3),
                                     data_format='channels_last', padding='same'))(net)
        net = BatchNormalization()(net)
        net = TimeDistributed(SwishLayer())(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # skip connection
        skip_16x16x128 = EncoderResidualLayer(depth=128, name="skip_16x16x128")(net)

        # 16x16x64
        net = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3),
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
        skip_8x8x256 = EncoderResidualLayer(depth=256, name="skip_8x8x256")(net)

        # 8x8x128
        net = TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3),
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
        skip_4x4x512 = EncoderResidualLayer(depth=512, name="skip_4x4x512")(net)

        # 4x4x256
        net = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3),
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
        mean = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                      padding='same', name="mean_convolution")(net)
        mean = MaxPool2D(pool_size=(2, 2),
                         name="mean_max_pooling")(mean)
        mean = Flatten(name="mean_flatten")(mean)
        mean = Dense(self.latent_size,
                     name="mean")(mean)
        stddev = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                        padding='same', name="stddev_convolution")(net)
        stddev = MaxPool2D(pool_size=(2, 2),
                           name="stddev_max_pooling")(stddev)
        stddev = Flatten(name="stddev_flatten")(stddev)
        stddev = Dense(self.latent_size,
                       name="stddev")(stddev)

        return input_layer, [mean, stddev, skip_4x4x512, skip_8x8x256, skip_16x16x128, skip_32x32x64, skip_64x64x32]

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class NVAEDecoder128(Architecture):
    def __init__(self,
                 frames_no=3,
                 alpha=10.0,
                 beta=10.0,
                 batch_size=4):
        self.alpha = alpha
        self.beta = beta
        self.mask_input_shape = (frames_no, 128, 128, 1)
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         frames_no=frames_no,
                         latent_size=1024)

    def layers(self):
        mean_input = Input(self.latent_size, self.batch_size, name="mean_input")
        stddev_input = Input(self.latent_size, self.batch_size, name="stddev_input")
        mask_input = Input(self.mask_input_shape, self.batch_size, name="mask_input")
        skip_4x4x512 = Input((4, 4, 512), self.batch_size, name="skip_4x4x512")
        skip_8x8x256 = Input((8, 8, 256), self.batch_size, name="skip_8x8x256")
        skip_16x16x128 = Input((16, 16, 128), self.batch_size, name="skip_16x16x128")
        skip_32x32x64 = Input((32, 32, 64), self.batch_size, name="skip_32x32x64")
        skip_64x64x32 = Input((64, 64, 32), self.batch_size, name="skip_64x64x32")

        ########################
        # Mask encoder network #
        ########################

        # 128x128x3
        mask_net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                          use_bias=False, data_format='channels_last',
                                          padding='same'))(mask_input)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                          use_bias=False, data_format='channels_last',
                                          padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 64x64x16
        mask_net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                          use_bias=False, data_format='channels_last',
                                          padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                          use_bias=False, data_format='channels_last',
                                          padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask64x64x32 = ConvLSTM2D(filters=32, kernel_size=(3, 3), data_format='channels_last',
                                  padding='same', return_sequences=False)(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 32x32x32
        mask_net = TimeDistributed(Conv2D(filters=64, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=64, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask32x32x64 = ConvLSTM2D(filters=64, kernel_size=(3, 3), data_format='channels_last',
                                  padding='same', return_sequences=False)(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 16x16x64
        mask_net = TimeDistributed(Conv2D(filters=128, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=128, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask16x16x128 = ConvLSTM2D(filters=128, kernel_size=(3, 3), data_format='channels_last',
                                   padding='same', return_sequences=False)(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 8x8x128
        mask_net = TimeDistributed(Conv2D(filters=256, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=256, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask8x8x256 = ConvLSTM2D(filters=256, kernel_size=(3, 3), data_format='channels_last',
                                 padding='same', return_sequences=False)(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 4x4x256
        mask_net = TimeDistributed(Conv2D(filters=512, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = TimeDistributed(Conv2D(filters=512, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask4x4x512 = ConvLSTM2D(filters=512, kernel_size=(3, 3), data_format='channels_last',
                                 padding='same', return_sequences=False)(mask_net)
        mask_net = TimeDistributed(MaxPooling2D(pool_size=2))(mask_net)
        mask_net = TimeDistributed(Dropout(self.dropout))(mask_net)

        # 2x2x512
        mask_net = TimeDistributed(Conv2D(filters=1024, kernel_size=3, use_bias=False,
                                          data_format='channels_last', padding='same'))(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = TimeDistributed(SwishLayer())(mask_net)
        mask_net = ConvLSTM2D(filters=1024, kernel_size=(3, 3), data_format='channels_last',
                              padding='same', return_sequences=False)(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = SwishLayer()(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask_net)
        # 1x1x1024
        mask_net = Dropout(self.dropout)(mask_net)
        mask_net = Flatten(name="mask_flatten")(mask_net)
        mask_net = Dense(self.latent_size, name="epsilon_dense")(mask_net)

        epsilon = EpsilonLayer(alpha=self.alpha, name="epsilon_layer")(mask_net)

        sample = SampleLayer(beta=self.beta, capacity=self.latent_size,
                             name="sampling_layer")([mean_input, stddev_input, epsilon])

        ###################
        # Decoder network #
        ###################
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latent_size))(sample)
        net = SwishLayer()(net)

        # 1x1x1024
        net = Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(net)
        # 2x2x1024
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=1024, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=1024, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 2x2x1024
        net = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(net)
        # 4x4x512

        # sampling
        decoded_4x4x1024 = concatenate([net, skip_4x4x512])
        mean_4x4x512 = NVAEResidualLayer(depth=512, name="mean_4x4x512")(decoded_4x4x1024)
        mean_4x4x512 = Flatten(data_format="channels_last")(mean_4x4x512)
        stddev_4x4x512 = NVAEResidualLayer(depth=512, name="stddev_4x4x512")(decoded_4x4x1024)
        stddev_4x4x512 = Flatten(data_format="channels_last")(stddev_4x4x512)
        epsilon_4x4x512 = EpsilonLayer(alpha=self.alpha, name="epsilon_4x4x512")(mask4x4x512)
        epsilon_4x4x512 = Flatten(data_format="channels_last")(epsilon_4x4x512)
        sample_4x4x512 = SampleLayer(beta=self.beta, capacity=512,
                                     name="sample_4x4x512")([mean_4x4x512, stddev_4x4x512, epsilon_4x4x512])
        sample_4x4x512 = Reshape((4, 4, 512))(sample_4x4x512)
        # 4x4x512
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=512, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 4x4x512
        net = concatenate([net, sample_4x4x512])
        # 4x4x1024
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=512, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 4x4x512
        net = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(net)
        # 8x8x256
        # sampling
        decoded_8x8x256 = concatenate([net, skip_8x8x256])
        mean_8x8x256 = NVAEResidualLayer(depth=256, name="mean_8x8x256")(decoded_8x8x256)
        mean_8x8x256 = Flatten(data_format="channels_last")(mean_8x8x256)
        stddev_8x8x256 = NVAEResidualLayer(depth=256, name="stddev_8x8x256")(decoded_8x8x256)
        stddev_8x8x256 = Flatten(data_format="channels_last")(stddev_8x8x256)
        epsilon_8x8x256 = EpsilonLayer(alpha=self.alpha, name="epsilon_8x8x256")(mask8x8x256)
        epsilon_8x8x256 = Flatten(data_format="channels_last")(epsilon_8x8x256)
        sample_8x8x256 = SampleLayer(beta=self.beta, capacity=256,
                                     name="sample_8x8x256")([mean_8x8x256, stddev_8x8x256, epsilon_8x8x256])
        sample_8x8x256 = Reshape((8, 8, 256))(sample_8x8x256)
        # 8x8x256
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=256, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 8x8x256
        net = concatenate([net, sample_8x8x256])
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=256, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 8x8x256
        net = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(net)
        # 16x16x128
        # sampling
        decoded_16x16x128 = concatenate([net, skip_16x16x128])
        mean_16x16x128 = NVAEResidualLayer(depth=128, name="mean_16x16x128")(decoded_16x16x128)
        mean_16x16x128 = Flatten(data_format="channels_last")(mean_16x16x128)
        stddev_16x16x128 = NVAEResidualLayer(depth=128, name="stddev_16x16x128")(decoded_16x16x128)
        stddev_16x16x128 = Flatten(data_format="channels_last")(stddev_16x16x128)
        epsilon_16x16x128 = EpsilonLayer(alpha=self.alpha, name="epsilon_16x16x128")(mask16x16x128)
        epsilon_16x16x128 = Flatten(data_format="channels_last")(epsilon_16x16x128)
        sample_16x16x128 = SampleLayer(beta=self.beta, capacity=128,
                                       name="sample_16x16x128")([mean_16x16x128, stddev_16x16x128, epsilon_16x16x128])
        sample_16x16x128 = Reshape((16, 16, 128))(sample_16x16x128)
        # 16x16x256
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=128, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 16x16x128
        net = concatenate([net, sample_16x16x128])
        # 16x16x256
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=128, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 16x16x128
        net = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(net)
        # 32x32x64
        # sampling
        decoded_32x32x64 = concatenate([net, skip_32x32x64])
        mean_32x32x64 = NVAEResidualLayer(depth=64, name="mean_32x32x64")(decoded_32x32x64)
        mean_32x32x64 = Flatten(data_format="channels_last")(mean_32x32x64)
        stddev_32x32x64 = NVAEResidualLayer(depth=64, name="stddev_32x32x64")(decoded_32x32x64)
        stddev_32x32x64 = Flatten(data_format="channels_last")(stddev_32x32x64)
        epsilon_32x32x64 = EpsilonLayer(alpha=self.alpha, name="epsilon_32x32x64")(mask32x32x64)
        epsilon_32x32x64 = Flatten(data_format="channels_last")(epsilon_32x32x64)
        sample_32x32x64 = SampleLayer(beta=self.beta, capacity=64,
                                      name="sample_32x32x64")([mean_32x32x64, stddev_32x32x64, epsilon_32x32x64])
        sample_32x32x64 = Reshape((32, 32, 64))(sample_32x32x64)
        # 32x32x128
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=64, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 32x32x64
        net = concatenate([net, sample_32x32x64])
        net = Dropout(self.dropout)(net)
        # 32x32x128
        net = Conv2D(filters=64, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 32x32x64
        net = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net)
        # 64x64x32
        # sampling
        decoded_64x64x32 = concatenate([net, skip_64x64x32])
        mean_64x64x32 = NVAEResidualLayer(depth=32, name="mean_64x64x32")(decoded_64x64x32)
        mean_64x64x32 = Flatten(data_format="channels_last")(mean_64x64x32)
        stddev_64x64x32 = NVAEResidualLayer(depth=32, name="stddev_64x64x32")(decoded_64x64x32)
        stddev_64x64x32 = Flatten(data_format="channels_last")(stddev_64x64x32)
        epsilon_64x64x32 = EpsilonLayer(alpha=self.alpha, name="epsilon_64x64x32")(mask64x64x32)
        epsilon_64x64x32 = Flatten(data_format="channels_last")(epsilon_64x64x32)
        sample_64x64x32 = SampleLayer(beta=self.beta, capacity=32,
                                      name="sample_64x64x32")([mean_64x64x32, stddev_64x64x32, epsilon_64x64x32])
        sample_64x64x32 = Reshape((64, 64, 32))(sample_64x64x32)
        # 64x64x64
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=32, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        # 64x64x32
        net = concatenate([net, sample_64x64x32])
        net = Dropout(self.dropout)(net)
        # 64x64x64
        net = Conv2D(filters=32, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)

        # 64x64x32
        net = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)
        # 128x128x16
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=16, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = SwishLayer()(net)
        net = Conv2D(filters=3, kernel_size=3, use_bias=False,
                     data_format='channels_last', padding='same')(net)
        return [mean_input, stddev_input, mask_input,
                skip_4x4x512, skip_8x8x256, skip_16x16x128, skip_32x32x64, skip_64x64x32], net

    def Build(self):
        decoder_input, decoder_output = self.layers()
        return Model(inputs=decoder_input, outputs=decoder_output)


class NVAEAutoEncoder128(object):
    def __init__(self, batch_size=16, alpha=5.0, beta=5.0,
                 encoder_frames_no=30, decoder_frames_no=3):
        self.latent_size = 1024
        self.batch_size = batch_size
        self.encoder_frames_no = encoder_frames_no
        self.decoder_frames_no = decoder_frames_no
        self.encoder = NVAEEncoder128(batch_size=batch_size,
                                      frames_no=encoder_frames_no)
        self.encoder_model = self.encoder.Build()
        self.decoder = NVAEDecoder128(batch_size=batch_size,
                                      frames_no=decoder_frames_no,
                                      alpha=alpha,
                                      beta=beta)
        self.decoder_model = self.decoder.Build()
        self.model = self.Build()

    def Build(self):
        sequence_input = Input((self.encoder_frames_no, 128, 128, 3), self.batch_size)
        detail_input = Input((128, 128, 3), self.batch_size)
        mask_input = Input((self.decoder_frames_no, 128, 128, 1), self.batch_size)
        encoder_output = self.encoder_model(sequence_input)
        mean, stddev, skip_4x4x512, skip_8x8x256, skip_16x16x128, skip_32x32x64, skip_64x64x32 = tuple(encoder_output)
        decoder_output = self.decoder_model([mean, stddev, mask_input,
                                             skip_4x4x512, skip_8x8x256, skip_16x16x128, skip_32x32x64, skip_64x64x32])
        net = DummyMaskLayer()(decoder_output)
        return Model(inputs=[sequence_input, detail_input, mask_input], outputs=net)

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
                                      decoder_frames_no=3,
                                      alpha=5.0,
                                      beta=5.0)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
