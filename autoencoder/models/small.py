from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape, MaxPool3D, UpSampling2D

from autoencoder.models.utils import LSTMConvBnRelu, SampleLayer, SequencesToBatchesLayer, ConvBnRelu, DummyMaskLayer


class Architecture(object):
    """
    generic architecture template
    """

    def __init__(self, inputShape=None, batchSize=None, latentSize=None, framesNo=None):
        """
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        """
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize
        self.framesNo = framesNo

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class LSTMEncoder32(Architecture):
    """
        This encoder predicts distributions then randomly samples them.
        Regularization may be applied to the latent space output

        a simple, fully convolutional architecture inspried by
            pjreddie's darknet architecture
        https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
        """

    def __init__(self, inputShape=(32, 32, 3), batchSize=1, framesNo=8,
                 latentSize=512, latentConstraints='bvae', beta=100., capacity=512.,
                 randomSample=True):
        """
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        """
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        self.framesNo = framesNo
        self.realInputShape = (framesNo,) + inputShape
        super(LSTMEncoder32, self).__init__(inputShape, batchSize, latentSize, framesNo)

    def layers(self):
        # create the input layer for feeding the network
        inLayer = Input(self.realInputShape, self.batchSize)
        net = LSTMConvBnRelu(lstm_kernels=[(3, 64)], conv_kernels=[(3, 128), (1, 64), (3, 128)])(inLayer)
        net = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(net)

        net = LSTMConvBnRelu(lstm_kernels=[(3, 128)], conv_kernels=[(3, 256), (1, 128), (3, 256)])(net)
        net = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(net)

        net = LSTMConvBnRelu(lstm_kernels=[(3, 256)], conv_kernels=[(3, 512), (1, 256), (3, 512)])(net)
        net = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(net)
        # reshape for sequence removal
        # there should be only one element of sequence at this point so it is just dimension reduction
        net = SequencesToBatchesLayer()(net)
        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                             self.latentCapacity, self.randomSample)([mean, stddev])
        return inLayer, sample

    def Build(self):
        inputs, outputs = self.layers()
        return Model(inputs=inputs, outputs=outputs)


class LSTMDecoder32(Architecture):
    def __init__(self, inputShape=(32, 32, 3), batchSize=1, latentSize=512):
        super(LSTMDecoder32, self).__init__(inputShape, batchSize, latentSize)

    def layers(self, inLayer):
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)

        net = UpSampling2D(size=(4, 4))(net)
        net = ConvBnRelu(conv_kernels=[(3, 256), (1, 128), (3, 256)])(net)

        net = UpSampling2D(size=(2, 2))(net)
        net = ConvBnRelu(conv_kernels=[(3, 128), (1, 64), (3, 128)])(net)

        net = UpSampling2D(size=(2, 2))(net)
        net = ConvBnRelu(conv_kernels=[(3, 64), (1, 32), (3, 64)])(net)

        net = UpSampling2D(size=(2, 2))(net)
        net = ConvBnRelu(conv_kernels=[(3, 32), (1, 16), (3, 32)])(net)

        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                     padding='same')(net)
        return inLayer, net

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        inputs, outputs = self.layers(inLayer)
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
    d19e = LSTMEncoder32()
    d19e.model.summary()
    d19d = LSTMDecoder32()
    d19d.model.summary()
    auto_encoder = VariationalAutoEncoder(d19e, d19d)
    auto_encoder.summary()


if __name__ == '__main__':
    test_summary()
