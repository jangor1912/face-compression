from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import MaxPool2D, Conv2D, GlobalAveragePooling2D, Reshape, UpSampling2D

from autoencoder.models.utils import LSTMConvBnRelu, SampleLayer


class Architecture(object):
    """
    generic architecture template
    """

    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
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

    def __init__(self, inputShape=(32, 32, 3), batchSize=16,
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
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the network
        inLayer = Input(self.inputShape, self.batchSize)
        net = LSTMConvBnRelu(lstm_kernels=[(3, 64)], conv_kernels=[(3, 128), (3, 128)])(inLayer)
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = LSTMConvBnRelu(lstm_kernels=[(3, 128)], conv_kernels=[(3, 256), (3, 256)])(net)
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = LSTMConvBnRelu(lstm_kernels=[(3, 256)], conv_kernels=[(3, 512), (3, 512)])(net)
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                             self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)


class LSTMDecoder32(Architecture):
    def __init__(self, inputShape=(32, 32, 3), batchSize=16, latentSize=512):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0] // 32, self.inputShape[1] // 32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)
        net = ConvBnLRelu(128, kernelSize=1)(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(32, kernelSize=3)(net)
        # net = ConvBnLRelu(3, kernelSize=1)(net)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                     padding='same')(net)

        return Model(inLayer, net)


class AutoEncoder(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder.model
        self.decoder = decoder.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))
