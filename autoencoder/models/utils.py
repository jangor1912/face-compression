"""
model_utils.py
contains custom layers, etc. for building mdoels.

created by shadySource

THE UNLICENSE
"""
import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, \
    Reshape, TimeDistributed, add, SeparableConv2D, ConvLSTM2D


class SwishLayer(K.layers.Layer):
    @staticmethod
    def swish(x, beta=1):
        return x * K.backend.sigmoid(beta * x)

    def call(self, x, **kwargs):
        x = Activation(self.swish)(x)
        return x


class ConvBnRelu(K.Model):

    def __init__(self, conv_kernels, stride=1, data_format='NHWC'):
        super(ConvBnRelu, self).__init__()
        data_format_keras = 'channels_last'
        channel_axis = 1 if data_format[1] == 'C' else -1
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.stride = stride

        for kxy, filters in conv_kernels:
            self.Conv.append(K.layers.Conv2D(filters=filters, kernel_size=kxy, strides=self.stride, use_bias=True,
                                             data_format=data_format_keras, padding='same'))
            self.BN.append(K.layers.BatchNormalization(axis=channel_axis))
            self.LReLU.append(K.layers.LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        activ = inputs  # set input to for loop
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(activ)
            bn = bn_layer(conv)
            activ = lrelu_layer(bn)
        return activ


class LSTMConvBnRelu(K.Model):

    def __init__(self, lstm_kernels, conv_kernels, stride=1, data_format='NHWC'):
        super(LSTMConvBnRelu, self).__init__()
        data_format_keras = 'channels_last'
        channel_axis = 1 if data_format[1] == 'C' else -1
        self.ConvLSTM = []
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.stride = stride

        for kernel_size, filters in lstm_kernels:
            self.ConvLSTM.append(K.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                                     padding='same', data_format=data_format_keras,
                                                     return_sequences=True, stateful=True))

        for kxy, filters in conv_kernels:
            self.Conv.append(K.layers.Conv2D(filters=filters, kernel_size=kxy, strides=self.stride, use_bias=True,
                                             data_format=data_format_keras, padding='same'))
            self.BN.append(K.layers.BatchNormalization(axis=channel_axis))
            self.LReLU.append(K.layers.LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        convlstm = inputs
        for conv_lstm_layer in self.ConvLSTM:
            convlstm = conv_lstm_layer(convlstm)

        orig_shape = convlstm.shape

        conv_input = tf.reshape(convlstm, [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]])
        activ = conv_input  # set input to for loop
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(activ)
            bn = bn_layer(conv)
            activ = lrelu_layer(bn)
        out_shape = activ.shape
        activ_down = tf.reshape(activ, [orig_shape[0], orig_shape[1], out_shape[1], out_shape[2], out_shape[3]])
        return activ_down

    def reset_states_per_batch(self, is_last_batch):
        batch_size = is_last_batch.shape[0]
        is_last_batch = tf.reshape(is_last_batch, [batch_size, 1, 1, 1])
        for convlstm_layer in self.ConvLSTM:
            cur_state = convlstm_layer.states
            new_states = (cur_state[0] * is_last_batch, cur_state[1] * is_last_batch)
            convlstm_layer.states[0].assign(new_states[0])
            convlstm_layer.states[1].assign(new_states[1])

    def get_states(self):
        states = []
        for convlstm_layer in self.ConvLSTM:
            state = convlstm_layer.states
            states.append([s.numpy() if s is not None else s for s in state])

        return states

    def set_states(self, states):
        for convlstm_layer, state in zip(self.ConvLSTM, states):
            if None is state[0]:
                state = None
            convlstm_layer.reset_states(state)


class SampleLayer(K.layers.Layer):
    """
    Keras Layer to grab a random sample from a distribution (by multiplication)
    Computes "(normal)*stddev + mean" for the vae sampling operation
    (written for tf backend)

    Additionally,
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization.

    call:
        pass in mean then stddev layers to sample from the distribution
        ex.
            sample = SampleLayer('bvae', 16)([mean, stddev])
    """

    def __init__(self,
                 beta=0.2,
                 epsilon_sequence=False,
                 relative=False,
                 **kwargs):
        self.beta = beta
        self.epsilon_sequence = epsilon_sequence
        self.shape = None
        self.relative = relative
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        self.shape = input_shape[0]

        super(SampleLayer, self).build(input_shape)  # needed for layers

    def call(self, x, **kwargs):
        mean = x[0]
        stddev = x[1]
        epsilon = x[2]

        if self.relative:
            # This assumes that previous_mean and previous_stddev exist within input
            previous_mean = x[3]
            previous_stddev = x[4]

            delta_mean = mean - previous_mean
            delta_stddev = stddev - previous_stddev

            mean = mean + delta_mean
            stddev = stddev * delta_stddev

        # kl divergence:
        latent_loss = -0.5 * K.backend.mean(1 + stddev
                                            - K.backend.square(mean)
                                            - K.backend.exp(stddev), axis=-1)
        latent_loss = latent_loss * self.beta
        self.add_loss(latent_loss, inputs=True)

        # epsilon = self.epsilon or K.backend.random_normal(shape=self.shape,
        #                                                   mean=0., stddev=1.)
        # 'reparameterization trick':
        if self.epsilon_sequence:
            mean = K.backend.repeat(mean, epsilon.shape[1])
            stddev = K.backend.repeat(stddev, epsilon.shape[1])
        return mean + K.backend.exp(stddev * 0.5) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class EpsilonLayer(K.layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(EpsilonLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        mean = K.backend.zeros(x.shape)
        stddev = x
        # kl divergence:
        latent_loss = -0.5 * K.backend.mean(1 + stddev
                                            - K.backend.square(mean)
                                            - K.backend.exp(stddev), axis=-1)
        latent_loss = latent_loss * self.alpha
        self.add_loss(latent_loss, inputs=True)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class SELayer(K.layers.Layer):
    def __init__(self, depth, **kwargs):
        self.depth = depth
        super(SELayer, self).__init__(**kwargs)

    def squeeze_excite_block(self, input_tensor, ratio=16):
        """ Create a channel-wise squeeze-excite block
        Args:
            input_tensor: input Keras tensor
            ratio: number of output filters
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        """
        init = input_tensor
        channel_axis = -1  # Assumption that the channels are last
        filters = init._shape_val[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = K.layers.multiply([init, se])
        return x

    def spatial_squeeze_excite_block(self, input_tensor):
        """ Create a spatial squeeze-excite block
        Args:
            input_tensor: input Keras tensor
        Returns: a Keras tensor
        References
        -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
        """

        se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                    kernel_initializer='he_normal')(input_tensor)

        x = K.layers.multiply([input_tensor, se])
        return x

    def channel_spatial_squeeze_excite(self, input_tensor, ratio=16):
        """ Create a spatial squeeze-excite block
        Args:
            input_tensor: input Keras tensor
            ratio: number of output filters
        Returns: a Keras tensor
        References
        -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
        """

        cse = self.squeeze_excite_block(input_tensor, ratio)
        sse = self.spatial_squeeze_excite_block(input_tensor)

        x = add([cse, sse])
        return x

    def call(self, x, **kwargs):
        x = self.channel_spatial_squeeze_excite(x, ratio=self.depth)
        return x


class EncoderResidualLayer(K.layers.Layer):
    def __init__(self, depth, **kwargs):
        self.depth = depth
        super(EncoderResidualLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        x = BatchNormalization()(x)
        x = TimeDistributed(SwishLayer())(x)
        x = ConvLSTM2D(filters=self.depth, kernel_size=(3, 3), data_format='channels_last',
                       padding='same', return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = SwishLayer()(x)
        x = Conv2D(filters=self.depth, kernel_size=3,
                   use_bias=False, data_format='channels_last',
                   padding='same')(x)
        x = SELayer(depth=self.depth)(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3], self.depth


class MaskResidualLayer(K.layers.Layer):
    def __init__(self, depth, **kwargs):
        self.depth = depth
        super(MaskResidualLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        x = BatchNormalization()(x)
        x = SwishLayer()(x)
        x = Conv2D(filters=self.depth, kernel_size=3,
                   use_bias=False, data_format='channels_last',
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = SwishLayer()(x)
        x = Conv2D(filters=self.depth, kernel_size=3,
                   use_bias=False, data_format='channels_last',
                   padding='same')(x)
        x = SELayer(depth=self.depth)(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.depth


class NVAEResidualLayer(K.layers.Layer):
    def __init__(self, depth, **kwargs):
        self.depth = depth
        super(NVAEResidualLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        x = BatchNormalization()(x)
        x = Conv2D(filters=self.depth, kernel_size=1,
                   use_bias=False, data_format='channels_last',
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = SwishLayer()(x)
        x = SeparableConv2D(filters=self.depth, kernel_size=5,
                            use_bias=False, data_format='channels_last',
                            padding='same')(x)
        x = BatchNormalization()(x)
        x = SwishLayer()(x)
        x = Conv2D(filters=self.depth, kernel_size=1,
                   use_bias=False, data_format='channels_last',
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = SELayer(depth=self.depth)(x)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.depth


class SequencesToBatchesLayer(K.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SequencesToBatchesLayer, self).__init__(*args, **kwargs)
        self._input_shape = None

    def build(self, input_shape):
        self._input_shape = input_shape
        super(SequencesToBatchesLayer, self).build(input_shape)

    def call(self, x):
        net = tf.reshape(x, [self._input_shape[0] * self._input_shape[1], self._input_shape[2],
                             self._input_shape[3], self._input_shape[4]])
        return net

    def compute_output_shape(self, input_shape):
        return [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]]


class BatchesToSequencesLayer(K.layers.Layer):
    def __init__(self, previous_shape, *args, **kwargs):
        super(BatchesToSequencesLayer, self).__init__(*args, **kwargs)
        self._input_shape = None
        self.previous_shape = previous_shape

    def build(self, input_shape):
        self._input_shape = input_shape
        super(BatchesToSequencesLayer, self).build(input_shape)

    def call(self, x):
        input_shape = x.shape
        net = tf.reshape(x, [self.previous_shape[0], self.previous_shape[1],
                             input_shape[1], input_shape[2], input_shape[3]])
        return net

    def compute_output_shape(self, input_shape):
        return [self.previous_shape[0], self.previous_shape[1],
                input_shape[1], input_shape[2], input_shape[3]]


class DummyMaskLayer(K.layers.Layer):
    def call(self, x):
        input_shape = x.shape
        net = tf.expand_dims(x, 1)
        to_append = tf.zeros(input_shape[1:])  # skipping batch dimension
        net = tf.map_fn(lambda batch: tf.concat((batch, [to_append]), axis=0), net)
        return net

    def compute_output_shape(self, input_shape):
        return [input_shape[0], 2, input_shape[1],
                input_shape[2], input_shape[3]]
