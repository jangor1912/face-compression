from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, Reshape, UpSampling2D, MaxPool2D, Dense, BatchNormalization, \
    LeakyReLU, Flatten, \
    ConvLSTM2D, TimeDistributed, MaxPooling2D, Dropout, MaxPooling3D, Conv2DTranspose

from autoencoder.models.utils import SampleLayer, DummyMaskLayer, \
    EpsilonLayer


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
        self.leak = 0.2

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class LSTMEncoder128(Architecture):
    def __init__(self, batch_size=4):
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         latent_size=1024,
                         frames_no=16)

    def layers(self):
        input_layer = Input(self.real_input_shape, self.batch_size)

        # 16x128x128x3
        net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                     use_bias=True, data_format='channels_last',
                                     padding='same'))(input_layer)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=self.leak))(net)
        net = TimeDistributed(Conv2D(filters=16, kernel_size=3,
                                     use_bias=True, data_format='channels_last',
                                     padding='same'))(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=self.leak))(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 16x64x64x16
        net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                     use_bias=True, data_format='channels_last',
                                     padding='same'))(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=self.leak))(net)
        net = TimeDistributed(Conv2D(filters=32, kernel_size=3,
                                     use_bias=True, data_format='channels_last',
                                     padding='same'))(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=self.leak))(net)
        net = TimeDistributed(Dropout(self.dropout))(net)
        net = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(net)

        # 16x32x32x32
        net = ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Dropout(self.dropout)(net)
        net = MaxPooling3D(pool_size=(2, 2, 2))(net)

        # 8x16x16x64
        net = ConvLSTM2D(filters=128, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = ConvLSTM2D(filters=128, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Dropout(self.dropout)(net)
        net = MaxPooling3D(pool_size=(2, 2, 2))(net)

        # 4x8x8x128
        net = ConvLSTM2D(filters=256, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = ConvLSTM2D(filters=256, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Dropout(self.dropout)(net)
        net = MaxPooling3D(pool_size=(2, 2, 2))(net)

        # 2x4x4x256
        net = ConvLSTM2D(filters=512, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = ConvLSTM2D(filters=512, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Dropout(self.dropout)(net)
        net = MaxPooling3D(pool_size=(2, 2, 2))(net)

        # 1x2x2x512
        # reshape for sequence removal
        # there should be only one element of sequence at this point so it is just dimension reduction
        net = Reshape((2, 2, 512))(net)
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

        return input_layer, [mean, stddev]

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class TextureEncoder128(Architecture):
    def __init__(self, batch_size=4, input_shape=(128, 128, 3)):
        super().__init__(input_shape=input_shape,
                         batch_size=batch_size,
                         latent_size=1024)
        self.c128x128x16 = None
        self.c64x64x32 = None
        self.c32x32x64 = None
        self.c16x16x128 = None
        self.c8x8x256 = None
        self.c4x4x512 = None
        self.c2x2x1024 = None

    def layers(self):
        texture_input = Input(self.input_shape, self.batch_size)

        # 128x128x3
        net = Conv2D(filters=16, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(texture_input)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=16, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c128x128x16 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c128x128x16)
        net = Dropout(self.dropout)(net)

        # 64x64x16
        net = Conv2D(filters=32, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=32, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c64x64x32 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c64x64x32)
        net = Dropout(self.dropout)(net)

        # 32x32x32
        net = Conv2D(filters=64, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=64, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c32x32x64 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c32x32x64)
        net = Dropout(self.dropout)(net)

        # 16x16x64
        net = Conv2D(filters=128, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=128, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c16x16x128 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c16x16x128)
        net = Dropout(self.dropout)(net)

        # 8x8x128
        net = Conv2D(filters=256, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=256, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c8x8x256 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c8x8x256)
        net = Dropout(self.dropout)(net)

        # 4x4x256
        net = Conv2D(filters=512, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=512, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c4x4x512 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c4x4x512)
        net = Dropout(self.dropout)(net)

        # 2x2x512
        net = Conv2D(filters=1024, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=1024, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        self.c2x2x1024 = LeakyReLU(alpha=self.leak)(net)
        net = MaxPooling2D(pool_size=2)(self.c2x2x1024)
        net = Dropout(self.dropout)(net)

        # 1x1x1024
        return texture_input, net

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class LSTMDecoder128(Architecture):
    def __init__(self,
                 alpha=10.0,
                 beta=10.0,
                 batch_size=4):
        self.alpha = alpha
        self.beta = beta
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         latent_size=1024,
                         frames_no=8)

    def layers(self, mean, stddev, detail_input, mask_input):
        ##########################
        # Detail encoder network #
        ##########################

        # 128x128x3
        detail_net = Conv2D(filters=16, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_input)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=16, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail128x128x16 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail128x128x16)
        detail_net = Dropout(self.dropout)(detail_net)

        # 64x64x16
        detail_net = Conv2D(filters=32, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=32, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail64x64x32 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail64x64x32)
        detail_net = Dropout(self.dropout)(detail_net)

        # 32x32x32
        detail_net = Conv2D(filters=64, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=64, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail32x32x64 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail32x32x64)
        detail_net = Dropout(self.dropout)(detail_net)

        # 16x16x64
        detail_net = Conv2D(filters=128, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=128, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail16x16x128 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail16x16x128)
        detail_net = Dropout(self.dropout)(detail_net)

        # 8x8x128
        detail_net = Conv2D(filters=256, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=256, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail8x8x256 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail8x8x256)
        detail_net = Dropout(self.dropout)(detail_net)

        # 4x4x256
        detail_net = Conv2D(filters=512, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=512, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail4x4x512 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail4x4x512)
        detail_net = Dropout(self.dropout)(detail_net)

        # 2x2x512
        detail_net = Conv2D(filters=1024, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail_net = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = Conv2D(filters=1024, kernel_size=3,
                            use_bias=True, data_format='channels_last', padding='same')(detail_net)
        detail_net = BatchNormalization()(detail_net)
        detail2x2x1024 = LeakyReLU(alpha=self.leak)(detail_net)
        detail_net = MaxPooling2D(pool_size=2)(detail2x2x1024)
        # 1x1x1024
        detail_net = Dropout(self.dropout)(detail_net)

        ########################
        # Mask encoder network #
        ########################

        # 128x128x3
        mask_net = Conv2D(filters=16, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_input)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=16, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask128x128x16 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask128x128x16)
        mask_net = Dropout(self.dropout)(mask_net)

        # 64x64x16
        mask_net = Conv2D(filters=32, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=32, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask64x64x32 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask64x64x32)
        mask_net = Dropout(self.dropout)(mask_net)

        # 32x32x32
        mask_net = Conv2D(filters=64, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=64, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask32x32x64 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask32x32x64)
        mask_net = Dropout(self.dropout)(mask_net)

        # 16x16x64
        mask_net = Conv2D(filters=128, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=128, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask16x16x128 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask16x16x128)
        mask_net = Dropout(self.dropout)(mask_net)

        # 8x8x128
        mask_net = Conv2D(filters=256, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=256, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask8x8x256 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask8x8x256)
        mask_net = Dropout(self.dropout)(mask_net)

        # 4x4x256
        mask_net = Conv2D(filters=512, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=512, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask4x4x512 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask4x4x512)
        mask_net = Dropout(self.dropout)(mask_net)

        # 2x2x512
        mask_net = Conv2D(filters=1024, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = Conv2D(filters=1024, kernel_size=3,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask2x2x1024 = LeakyReLU(alpha=self.leak)(mask_net)
        mask_net = MaxPooling2D(pool_size=2)(mask2x2x1024)
        # 1x1x1024
        mask_net = Dropout(self.dropout)(mask_net)
        mask_net = Flatten(name="mask_flatten")(mask_net)
        mask_net = Dense(self.latent_size, name="epsilon_input")(mask_net)
        epsilon = EpsilonLayer(alpha=self.alpha, name="epsilon")(mask_net)

        sample = SampleLayer(beta=self.beta, capacity=self.latent_size,
                             name="sampling_layer")([mean, stddev, epsilon])

        ###################
        # Decoder network #
        ###################

        net = Dense(self.latent_size, activation='relu')(sample)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latent_size))(net)

        # 1x1x1024
        net = Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same')(net)
        # 2x2x1024
        net = concatenate([net, detail2x2x1024, mask2x2x1024])
        # 2x2x3072
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=1024, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=1024, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 2x2x1024
        net = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(net)
        # 4x4x512
        net = concatenate([net, detail4x4x512, mask4x4x512])
        # 4x4x1536
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=512, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=512, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 4x4x512
        net = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(net)
        # 8x8x256
        net = concatenate([net, detail8x8x256, mask8x8x256])
        # 8x8x768
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=256, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=256, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 8x8x256
        net = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(net)
        # 16x16x128
        net = concatenate([net, detail16x16x128, mask16x16x128])
        # 16x16x384
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=128, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=128, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 16x16x128
        net = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(net)
        # 32x32x64
        net = concatenate([net, detail32x32x64, mask32x32x64])
        # 32x32x192
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=64, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=64, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 32x32x64
        net = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(net)
        # 64x64x32
        net = concatenate([net, detail64x64x32, mask64x64x32])
        # 64x64x96
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=32, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=32, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 64x64x32
        net = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(net)
        # 128x128x16
        net = concatenate([net, detail128x128x16, mask128x128x16])
        # 128x128x48
        net = Dropout(self.dropout)(net)
        net = Conv2D(filters=16, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)
        net = Conv2D(filters=16, kernel_size=3,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=self.leak)(net)

        # 128x128x16
        net = Conv2D(filters=self.input_shape[-1], kernel_size=(1, 1),
                     padding='same')(net)
        return [mean, stddev, detail_input, mask_input], net

    def Build(self):
        mean_input = Input(self.latent_size, self.batch_size)
        stddev_input = Input(self.latent_size, self.batch_size)
        detail_input = Input((128, 128, 3), self.batch_size,
                             name="detail_input")
        mask_input = Input((128, 128, 1), self.batch_size,
                           name="mask_input")
        decoder_input, decoder_output = self.layers(mean_input, stddev_input,
                                                    detail_input, mask_input)
        return Model(inputs=decoder_input, outputs=decoder_output)


class VariationalAutoEncoder128(object):
    def __init__(self, batch_size=4, alpha=10.0, beta=10.0):
        self.latent_size = 1024
        self.batch_size = batch_size
        self.sequence_encoder = LSTMEncoder128(batch_size=batch_size)
        self.face_decoder = LSTMDecoder128(batch_size=batch_size,
                                           alpha=alpha,
                                           beta=beta)
        self.model = self.Build()

    def Build(self):
        sequence_inputs, sequence_outputs = self.sequence_encoder.layers()
        mean, stddev = sequence_outputs
        detail_input = Input((128, 128, 3), self.batch_size,
                             name="detail_input")
        mask_input = Input((128, 128, 1), self.batch_size,
                           name="mask_input")
        decoder_inputs, decoder_outputs = self.face_decoder.layers(mean, stddev,
                                                                   detail_input, mask_input)
        net = DummyMaskLayer()(decoder_outputs)
        return Model(inputs=[sequence_inputs, detail_input, mask_input], outputs=net)

    def summary(self):
        print("Model summary")
        self.model.summary()


def test_summary():
    sequence_encoder = LSTMEncoder128()
    sequence_encoder.model.summary()
    auto_encoder = VariationalAutoEncoder()
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
