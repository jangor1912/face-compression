from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, ZeroPadding2D

from autoencoder.models.blocks import encoder_convolutional_block, encoder_identity_block, encoder_residual_block, \
    mask_residual_block, mobnet_conv_block, mobnet_separable_conv_block
from autoencoder.models.utils import SimpleSamplingLayer


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
        self.mask2x2x1024 = None
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
        x = mobnet_separable_conv_block(x, num_filters=512, strides=1)
        x = mobnet_conv_block(x, num_filters=512, kernel_size=1)
        x = mobnet_separable_conv_block(x, num_filters=512, strides=2)
        # 2x2x512
        x = mobnet_conv_block(x, num_filters=1024, kernel_size=1)
        # 2x2x1024
        self.mask2x2x1024 = mask_residual_block(x, depth=1024)

        return mask_input, [self.mask32x32x64,
                            self.mask16x16x128,
                            self.mask8x8x256,
                            self.mask4x4x512,
                            self.mask2x2x1024]

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

        self.mean_2x2x1024 = None
        self.stddev_2x2x1024 = None

        super(NVAEEncoder64, self).__init__(input_shape=(128, 128, 3),
                                            batch_size=batch_size,
                                            frames_no=frames_no)

    def layers(self):
        input_layer = Input(self.real_input_shape, self.batch_size)
        # 64x64x3
        # Zero-Padding
        net = ZeroPadding2D((3, 3))(input_layer)

        # 70x70x3

        # Stage 0
        net = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', kernel_initializer=glorot_uniform(seed=0))(net)
        net = BatchNormalization(axis=3, name='bn_conv0')(net)
        net = Activation('relu')(net)

        # 32x32x64

        # Stage 1
        net = encoder_convolutional_block(net, f=3, filters=[64, 64, 128], stage=2, block='a', s=1)
        net = encoder_identity_block(net, 3, [64, 64, 128], stage=2, block='b')

        # 32x32x64
        self.mean_32x32x64 = encoder_residual_block(net, depth=64)
        self.stddev_32x32x64 = encoder_residual_block(net, depth=64)

        # Stage 2
        net = encoder_convolutional_block(net, f=3, filters=[64, 64, 128], stage=2, block='a', s=2)
        net = encoder_identity_block(net, 3, [64, 64, 128], stage=2, block='b')

        # 16x16x128
        self.mean_16x16x128 = encoder_residual_block(net, depth=128)
        self.stddev_16x16x128 = encoder_residual_block(net, depth=128)

        # Stage 3
        net = encoder_convolutional_block(net, f=3, filters=[128, 128, 256], stage=3, block='a', s=2)
        net = encoder_identity_block(net, 3, [128, 128, 256], stage=3, block='b')

        # 8x8x256
        self.mean_8x8x256 = encoder_residual_block(net, depth=256)
        self.stddev_8x8x256 = encoder_residual_block(net, depth=256)

        # Stage 4
        net = encoder_convolutional_block(net, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
        net = encoder_identity_block(net, 3, [256, 256, 512], stage=4, block='b')

        # 4x4x512
        self.mean_4x4x512 = encoder_residual_block(net, depth=512)
        self.stddev_4x4x512 = encoder_residual_block(net, depth=512)

        # Stage 5
        net = encoder_convolutional_block(net, f=3, filters=[512, 512, 1024], stage=5, block='a', s=2)
        net = encoder_identity_block(net, 3, [512, 512, 1024], stage=5, block='b')

        # 2x2x1024
        self.mean_2x2x1024 = encoder_residual_block(net, depth=1024)
        self.stddev_2x2x1024 = encoder_residual_block(net, depth=1024)

        return input_layer, [self.mean_2x2x1024, self.stddev_2x2x1024,
                             self.mean_4x4x512, self.stddev_4x4x512,
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
        mean_2x2x1024 = Input((2, 2, 1024), self.batch_size, name="mean_2x2x1024")
        stddev_2x2x1024 = Input((2, 2, 1024), self.batch_size, name="stddev_2x2x1024")
        mean_4x4x512 = Input((4, 4, 512), self.batch_size, name="mean_4x4x512")
        stddev_4x4x512 = Input((4, 4, 512), self.batch_size, name="stddev_4x4x512")
        mean_8x8x256 = Input((8, 8, 256), self.batch_size, name="mean_8x8x256")
        stddev_8x8x256 = Input((8, 8, 256), self.batch_size, name="stddev_8x8x256")
        mean_16x16x128 = Input((16, 16, 128), self.batch_size, name="mean_16x16x128")
        stddev_16x16x128 = Input((16, 16, 128), self.batch_size, name="stddev_16x16x128")
        mean_32x32x64 = Input((32, 32, 64), self.batch_size, name="mean_32x32x64")
        stddev_32x32x64 = Input((32, 32, 64), self.batch_size, name="stddev_32x32x64")

        # Mask encoder inputs
        mask2x2x1024 = Input((2, 2, 1024), self.batch_size, name="mask2x2x1024")
        mask4x4x512 = Input((4, 4, 512), self.batch_size, name="mask4x4x512")
        mask8x8x256 = Input((8, 8, 256), self.batch_size, name="mask8x8x256")
        mask16x16x128 = Input((16, 16, 128), self.batch_size, name="mask16x16x128")
        mask32x32x64 = Input((32, 32, 64), self.batch_size, name="mask32x32x64")

        ###################
        # Decoder network #
        ###################

        sample2x2x1024 = SimpleSamplingLayer()([mean_2x2x1024, stddev_2x2x1024, mask2x2x1024])
        net = sample2x2x1024
        # TODO finish it!

        return [mean_2x2x1024, stddev_2x2x1024,
                mean_4x4x512, stddev_4x4x512,
                mean_8x8x256, stddev_8x8x256,
                mean_16x16x128, stddev_16x16x128,
                mean_32x32x64, stddev_32x32x64,
                mask2x2x1024,
                mask4x4x512,
                mask8x8x256,
                mask16x16x128,
                mask32x32x64], net

    def Build(self):
        decoder_input, decoder_output = self.layers()
        return Model(inputs=decoder_input, outputs=decoder_output)
