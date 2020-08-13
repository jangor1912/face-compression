import numpy as np
from tensorflow.python import keras as k
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv2D, ConvLSTM2D, Dense, \
    DepthwiseConv2D, GlobalAveragePooling2D, Reshape, SeparableConv2D, TimeDistributed, add

from autoencoder.models.spectral_norm import ConvSN2D, ConvSN2DTranspose
from autoencoder.models.utils import SELayer, SwishLayer


def mobnet_separable_conv_block(net, num_filters, strides, alpha=1.0):
    net = DepthwiseConv2D(kernel_size=3, padding='same')(net)
    net = BatchNormalization(momentum=0.9997)(net)
    net = Activation('relu')(net)
    net = Conv2D(np.floor(num_filters * alpha), kernel_size=(1, 1), strides=strides,
                 use_bias=False, padding='same')(net)
    net = BatchNormalization(momentum=0.9997)(net)
    net = Activation('relu')(net)
    return net


def mobnet_conv_block(net, num_filters, kernel_size, strides=1, alpha=1.0):
    net = Conv2D(np.floor(num_filters * alpha), kernel_size=kernel_size, strides=strides,
                 use_bias=False, padding='same')(net)
    net = BatchNormalization(momentum=0.9997)(net)
    net = Activation('relu')(net)
    return net


def encoder_convolutional_block(net, f, filters, stage, block, s=2):
    # Defining name basis
    conv_name_base = 'enc_res' + str(stage) + block + '_branch'
    bn_name_base = 'enc_bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    net_shortcut = net

    #############
    # MAIN PATH #
    #############
    # First component of main path 
    net = TimeDistributed(ConvSN2D(filters=f1, kernel_size=(1, 1), strides=(s, s),
                                   padding='valid', name=conv_name_base + '2a',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2a')(net)
    net = SwishLayer()(net)

    # Second component of main path
    net = TimeDistributed(ConvSN2D(filters=f2, kernel_size=(f, f), strides=(1, 1),
                                   padding='same', name=conv_name_base + '2b',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2b')(net)
    net = SwishLayer()(net)

    # Third component of main path
    net = TimeDistributed(ConvSN2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', name=conv_name_base + '2c',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2c')(net)

    #################
    # SHORTCUT PATH #
    #################
    # net_shortcut = TimeDistributed(ConvSN2D(filters=f3, kernel_size=(1, 1), strides=(s, s),
    #                                       padding='valid', name=conv_name_base + '1',
    #                                       kernel_initializer=glorot_uniform(seed=0)))(net_shortcut)
    # net_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(net_shortcut)

    # nVAE implementation
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = SwishLayer()(net_shortcut)
    net_shortcut = TimeDistributed(ConvSN2D(filters=f3, kernel_size=3,
                                            use_bias=False, data_format='channels_last',
                                            padding='same'))(net_shortcut)
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = SwishLayer()(net_shortcut)
    net_shortcut = TimeDistributed(ConvSN2D(filters=f3, kernel_size=3,
                                            use_bias=False, data_format='channels_last',
                                            padding='same'))(net_shortcut)
    net_shortcut = TimeDistributed(SELayer(depth=f3))(net_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    net = Add()([net, net_shortcut])
    net = SwishLayer()(net)

    return net


def encoder_identity_block(net, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'enc_res' + str(stage) + block + '_branch'
    bn_name_base = 'enc_bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    net_shortcut = net

    # First component of main path
    net = TimeDistributed(ConvSN2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', name=conv_name_base + '2a',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2a')(net)
    net = SwishLayer()(net)

    # Second component of main path
    net = TimeDistributed(ConvSN2D(filters=f2, kernel_size=(f, f), strides=(1, 1),
                                   padding='same', name=conv_name_base + '2b',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2b')(net)
    net = SwishLayer()(net)

    # Third component of main path 
    net = TimeDistributed(ConvSN2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', name=conv_name_base + '2c',
                                   kernel_initializer=glorot_uniform(seed=0)))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2c')(net)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    net = Add()([net, net_shortcut])
    net = SwishLayer()(net)

    return net


def decoder_convolutional_block(net, f, filters, stage, block, s=2):
    # Defining name basis
    conv_name_base = 'dec_res' + str(stage) + block + '_branch'
    bn_name_base = 'dec_bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    f1, f2, f3 = filters

    # Save the input value
    net_shortcut = net

    #############
    # MAIN PATH #
    #############
    # First component of main path
    net = ConvSN2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                   padding='same', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2a')(net)
    net = SwishLayer()(net)

    # Second component of main path
    net = ConvSN2DTranspose(filters=f2, kernel_size=(f, f), strides=(s, s),
                            padding='same', name=conv_name_base + '2b',
                            kernel_initializer=glorot_uniform(seed=0))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2b')(net)
    net = SwishLayer()(net)

    # Third component of main path
    net = ConvSN2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                   padding='same', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(net)
    net = BatchNormalization(axis=-1, name=bn_name_base + '2c')(net)

    #################
    # SHORTCUT PATH #
    #################
    # net_shortcut = TimeDistributed(ConvSN2D(filters=f3, kernel_size=(1, 1), strides=(s, s),
    #                                       padding='valid', name=conv_name_base + '1',
    #                                       kernel_initializer=glorot_uniform(seed=0)))(net_shortcut)
    # net_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(net_shortcut)

    # nVAE implementation
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = ConvSN2D(filters=f3, kernel_size=1,
                            use_bias=False, data_format='channels_last',
                            padding='same')(net_shortcut)
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = SwishLayer()(net_shortcut)
    net_shortcut = SeparableConv2D(filters=f3, kernel_size=5,
                                   use_bias=False, data_format='channels_last',
                                   padding='same')(net_shortcut)
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = SwishLayer()(net_shortcut)
    net_shortcut = ConvSN2DTranspose(filters=f3, kernel_size=(3, 3), strides=(2, 2),
                                     use_bias=False, data_format='channels_last',
                                     padding='same')(net_shortcut)
    net_shortcut = BatchNormalization()(net_shortcut)
    net_shortcut = SELayer(depth=f3)(net_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    net = Add()([net, net_shortcut])
    net = SwishLayer()(net)

    return net


def se_block(net, depth, ratio=16):
    def squeeze_excite_block(input_tensor):
        """ Create a channel-wise squeeze-excite block
        Args:
            input_tensor: input Keras tensor
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        """
        init = input_tensor
        filters = depth
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = k.layers.multiply([init, se])
        return x

    def spatial_squeeze_excite_block(input_tensor):
        """ Create a spatial squeeze-excite block
        Args:
            input_tensor: input Keras tensor
        Returns: a Keras tensor
        References
        -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
        """

        se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                    kernel_initializer='he_normal')(input_tensor)

        x = k.layers.multiply([input_tensor, se])
        return x

    def channel_spatial_squeeze_excite(input_tensor):
        """ Create a spatial squeeze-excite block
        Args:
            input_tensor: input Keras tensor
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
        """

        cse = squeeze_excite_block(input_tensor)
        sse = spatial_squeeze_excite_block(input_tensor)

        x = add([cse, sse])
        return x

    net = channel_spatial_squeeze_excite(net)
    return net


def mask_residual_block(net, depth):
    net = BatchNormalization()(net)
    net = SwishLayer()(net)
    net = ConvSN2D(filters=depth, kernel_size=3,
                   use_bias=False, data_format='channels_last',
                   padding='same')(net)
    net = se_block(net, depth=depth, ratio=16)
    return net


def encoder_residual_block(net, depth):
    net = BatchNormalization()(net)
    net = TimeDistributed(SwishLayer())(net)
    net = ConvLSTM2D(filters=depth, kernel_size=(3, 3), data_format='channels_last',
                     padding='same', return_sequences=False)(net)
    net = BatchNormalization()(net)
    net = SwishLayer()(net)
    net = ConvSN2D(filters=depth, kernel_size=3,
                   use_bias=False, data_format='channels_last',
                   padding='same')(net)
    net = se_block(net, depth=depth, ratio=16)

    return net
