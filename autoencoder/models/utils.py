"""
model_utils.py
contains custom layers, etc. for building mdoels.

created by shadySource

THE UNLICENSE
"""
import tensorflow as tf
from tensorflow.python import keras as K


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

    def __init__(self, latent_regularizer='bvae',
                 beta=100., capacity=0.,
                 randomSample=True,
                 **kwargs):
        """
        args:
        ------
        latent_regularizer : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer,
            (Unused if 'bvae' not selected)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        ------
        ex.
            sample = SampleLayer('bvae', 16)([mean, stddev])
        """
        self.reg = latent_regularizer
        self.beta = beta
        self.capacity = capacity
        self.random = randomSample
        self.shape = None
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        self.shape = input_shape[0]

        super(SampleLayer, self).build(input_shape)  # needed for layers

    def call(self, x):
        if len(x) != 3:
            raise Exception('input layers must be a list: mean, stddev and epsilon')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [mean, stddev, epsilon]')

        mean = x[0]
        stddev = x[1]
        epsilon = x[2]

        if self.reg == 'bvae':
            # kl divergence:
            latent_loss = -0.5 * K.backend.mean(1 + stddev
                                                - K.backend.square(mean)
                                                - K.backend.exp(stddev), axis=-1)
            # use beta to force less usage of vector space:
            # also try to use <capacity> dimensions of the space:
            latent_loss = self.beta * K.backend.abs(latent_loss - self.capacity / self.shape.as_list()[1])
            self.add_loss(latent_loss, x)
        elif self.reg == 'vae':
            # kl divergence:
            latent_loss = -0.5 * K.backend.mean(1 + stddev
                                                - K.backend.square(mean)
                                                - K.backend.exp(stddev), axis=-1)
            self.add_loss(latent_loss, x)

        # epsilon = self.epsilon or K.backend.random_normal(shape=self.shape,
        #                                                   mean=0., stddev=1.)
        if self.random:
            # 'reparameterization trick':
            return mean + K.backend.exp(stddev) * epsilon
        else:  # do not perform random sampling, simply grab the impulse value
            return mean + 0 * stddev  # Keras needs the *0 so the gradinent is not None

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class EpsilonLayer(K.layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(EpsilonLayer, self).__init__(**kwargs)

    def call(self, x):
        # mse will keep epsilon as close to zero as possible
        # This will ensure that vector actually reminds gaussian random normal
        latent_loss = K.backend.mean(K.backend.square(x), axis=-1)
        latent_loss = self.alpha * latent_loss
        self.add_loss(latent_loss, x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


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
