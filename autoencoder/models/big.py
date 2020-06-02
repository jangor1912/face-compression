from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, Reshape, UpSampling2D, MaxPool2D, Dense, BatchNormalization, \
    LeakyReLU, Flatten, \
    ConvLSTM2D, TimeDistributed, MaxPooling2D, Dropout, MaxPooling3D

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

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class LSTMEncoder128(Architecture):
    def __init__(self, batch_size=4, alpha=10., beta=10.):
        self.alpha = alpha
        self.beta = beta
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         latent_size=1024,
                         frames_no=8)

    def layers(self):
        input_layer = Input(self.real_input_shape, self.batch_size)

        # 8x128x128x3
        net = ConvLSTM2D(filters=256, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(input_layer)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=128, kernel_size=(1, 1),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=256, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = MaxPooling3D(pool_size=(2, 4, 4), strides=(2, 4, 4))(net)

        # 4x32x32x256
        net = ConvLSTM2D(filters=512, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=256, kernel_size=(1, 1),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=512, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = MaxPooling3D(pool_size=(2, 4, 4), strides=(2, 4, 4))(net)

        # 2x8x8x512
        net = ConvLSTM2D(filters=1024, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=512, kernel_size=(1, 1),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = ConvLSTM2D(filters=1024, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(LeakyReLU(alpha=0.2))(net)
        net = TimeDistributed(Dropout(0.8))(net)
        net = MaxPooling3D(pool_size=(2, 4, 4), strides=(2, 4, 4))(net)

        # 1x2x2x1024
        # reshape for sequence removal
        # there should be only one element of sequence at this point so it is just dimension reduction
        net = Reshape(net.shape[2:])(net)
        # 2x2x1024
        # Second input - b&w mask based on which face pose should be generated
        # cut RGB since mask is black and white
        mask_input = Input(tuple(list(self.input_shape[:-1]) + [1]), self.batch_size)

        # 128x128x3
        mask_net = Conv2D(filters=256, kernel_size=3, strides=1,
                          use_bias=True, data_format='channels_last', padding='same')(mask_input)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=0.2)(mask_net)
        mask_net = Dropout(0.8)(mask_net)
        mask_net = MaxPooling2D(pool_size=4, strides=4)(mask_net)

        # 32x32x256
        mask_net = Conv2D(filters=512, kernel_size=3, strides=1,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=0.2)(mask_net)
        mask_net = Dropout(0.8)(mask_net)
        mask_net = MaxPooling2D(pool_size=4, strides=4)(mask_net)

        # 8x8x512
        mask_net = Conv2D(filters=1024, kernel_size=3, strides=1,
                          use_bias=True, data_format='channels_last', padding='same')(mask_net)
        mask_net = BatchNormalization()(mask_net)
        mask_net = LeakyReLU(alpha=0.2)(mask_net)
        mask_net = Dropout(0.8)(mask_net)
        mask_net = MaxPooling2D(pool_size=4, strides=4)(mask_net)

        # 2x2x1024
        mask_net = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                          padding='same', name="mask_net_convolution")(mask_net)
        mask_net = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             name="mask_net_max_pooling")(mask_net)
        mask_net = Flatten(name="mask_flatten")(mask_net)
        mask_net = Dense(self.latent_size, name="epsilon_input")(mask_net)
        epsilon = EpsilonLayer(alpha=self.alpha, name="epsilon")(mask_net)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                      padding='same', name="mean_convolution")(net)
        mean = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                         name="mean_max_pooling")(mean)
        mean = Flatten(name="mean_flatten")(mean)
        mean = Dense(self.latent_size,
                     name="mean")(mean)
        stddev = Conv2D(filters=self.latent_size, kernel_size=(1, 1),
                        padding='same', name="stddev_convolution")(net)
        stddev = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                           name="stddev_max_pooling")(stddev)
        stddev = Flatten(name="stddev_flatten")(stddev)
        stddev = Dense(self.latent_size,
                       name="stddev")(stddev)

        sample = SampleLayer(beta=self.beta, capacity=self.latent_size,
                             name="sampling_layer")([mean, stddev, epsilon])

        return [input_layer, mask_input], sample

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class LSTMDecoder128(Architecture):
    def __init__(self, batch_size=4):
        super().__init__(input_shape=(128, 128, 3),
                         batch_size=batch_size,
                         latent_size=1024,
                         frames_no=8)

    def layers(self, inLayer):
        net = Dense(self.latent_size, activation='relu')(inLayer)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latent_size))(net)

        net = UpSampling2D(size=(4, 4))(net)

        # 4x4x1024
        net = Conv2D(filters=512, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=256, kernel_size=1, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=512, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)

        net = UpSampling2D(size=(4, 4))(net)

        # 16x16x512
        net = Conv2D(filters=256, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=512, kernel_size=1, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=256, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)

        net = UpSampling2D(size=(4, 4))(net)

        # 64x64x256
        net = Conv2D(filters=128, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=256, kernel_size=1, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=128, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)

        net = UpSampling2D(size=(2, 2))(net)

        # 128x128x128
        net = Conv2D(filters=64, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=128, kernel_size=1, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)
        net = Conv2D(filters=64, kernel_size=3, strides=1,
                     use_bias=True, data_format='channels_last', padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.2)(net)
        net = Dropout(0.8)(net)

        net = Conv2D(filters=self.input_shape[-1], kernel_size=(1, 1),
                     padding='same')(net)
        return inLayer, net

    def Build(self):
        # input layer is from GlobalAveragePooling:
        input_layer = Input([self.latent_size], self.batch_size)
        inputs, outputs = self.layers(input_layer)
        return Model(inputs=inputs, outputs=outputs)


class VariationalAutoEncoder(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.model = self.Build()

    def Build(self):
        encoder_inputs, encoder_outputs = self.encoder.layers()
        decoder_inputs, decoder_outputs = self.decoder.layers(encoder_outputs)
        net = DummyMaskLayer()(decoder_outputs)
        return Model(inputs=encoder_inputs, outputs=net)

    def summary(self):
        print("Model summary")
        self.model.summary()


def test_summary():
    d19e = LSTMEncoder128()
    d19e.model.summary()
    d19d = LSTMDecoder128()
    d19d.model.summary()
    auto_encoder = VariationalAutoEncoder(d19e, d19d)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
