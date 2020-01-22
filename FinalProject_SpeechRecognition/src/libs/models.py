"""
Model definitions for simple speech recognition.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count,use_spectrogram=False):
    """Calculates common settings needed for all models.
  
    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.
  
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None,model_size_info=None):
    """Builds a model of the requested architecture compatible with the settings.
  
    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].
  
    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.
  
    See the implementations below for the possible model architectures that can be
    requested.
  
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      model_architecture: String specifying which kind of model to create.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.
  
    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
  
    Raises:
      Exception: If the architecture type isn't recognized.
    """
    if model_architecture == 'ds_cnn':
        return create_ds_cnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture == 'ds_cnn_spec':
        return create_ds_cnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture == 'c_rnn':
        return create_crnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    elif model_architecture=='c_rnn_spec':
        return create_crnn_model(fingerprint_input,model_settings,model_size_info,is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "ds_cnn", "ds_cnn_spec",' +
                        ' "c_rnn, or "c_rnn_spec"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
  
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

def create_ds_cnn_model(fingerprint_input, model_settings, model_size_info,
                        is_training):
    """Builds a model with depthwise separable convolutional neural network
    Model definition is based on https://arxiv.org/abs/1704.04861 and
    Tensorflow implementation: https://github.com/Zehaos/MobileNet
    model_size_info: defines number of layers, followed by the DS-Conv layer
      parameters in the order {number of conv features, conv filter height, 
      width and stride in y,x dir.} for each of the layers. 
    Note that first layer is always regular convolution, but the remaining 
      layers are all depthwise separable convolutions.
    """

    def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
  
    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers
    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.96,
                                updates_collections=None,
                                activation_fn=tf.nn.relu):
                for layer_no in range(0, num_layers):
                    if layer_no == 0:
                        net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                 [conv_kt[layer_no], conv_kf[layer_no]],
                                                 stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME',
                                                 scope='conv_1')
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    else:
                        net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                        kernel_size=[conv_kt[layer_no], conv_kf[layer_no]], \
                                                        stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                        sc='conv_ds_' + str(layer_no))
                    t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                    f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

                net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')

    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_crnn_model(fingerprint_input, model_settings,
                      model_size_info, is_training):
    """Builds a model with convolutional recurrent networks with GRUs
    Based on the model definition in https://arxiv.org/abs/1703.05390
    model_size_info: defines the following convolution layer parameters
        {number of conv features, conv filter height, width, stride in y,x dir.},
        followed by number of GRU layers and number of GRU cells per layer
    Optionally, the bi-directional GRUs and/or GRU with layer-normalization 
      can be explored.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = False

    # CNN part
    first_filter_count = model_size_info[0]
    first_filter_height = model_size_info[1]
    first_filter_width = model_size_info[2]
    first_filter_stride_y = model_size_info[3]
    first_filter_stride_x = model_size_info[4]

    first_weights = tf.get_variable('W', shape=[first_filter_height,
                                                first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    num_rnn_layers = model_size_info[5]
    RNN_units = model_size_info[6]
    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
    cell_fw = []
    cell_bw = []
    if layer_norm:
        for i in range(num_rnn_layers):
            cell_fw.append(LayerNormGRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(LayerNormGRUCell(RNN_units))
    else:
        for i in range(num_rnn_layers):
            cell_fw.append(tf.contrib.rnn.GRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(tf.contrib.rnn.GRUCell(RNN_units))

    if bidirectional:
        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, flow,
                                                           dtype=tf.float32)
        flow_dim = first_conv_output_height * RNN_units * 2
        flow = tf.reshape(outputs, [-1, flow_dim])
    else:
        cells = tf.contrib.rnn.MultiRNNCell(cell_fw)
        _, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
        flow_dim = RNN_units
        flow = last[-1]

    first_fc_output_channels = model_size_info[7]

    first_fc_weights = tf.get_variable('fcw', shape=[flow_dim,
                                                     first_fc_output_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())

    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
    if is_training:
        final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        final_fc_input = first_fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, label_count], stddev=0.01))

    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

class LayerNormGRUCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                 input_size=None, activation=math_ops.tanh,
                 layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                 dropout_keep_prob=1.0, dropout_prob_seed=None,
                 reuse=None):

        super(LayerNormGRUCell, self).__init__(_reuse=reuse)

        if input_size is not None:
            tf.logging.info("%s: The input_size parameter is deprecated.", self)

        self._num_units = num_units
        self._activation = activation
        self._forget_bias = forget_bias
        self._keep_prob = dropout_keep_prob
        self._seed = dropout_prob_seed
        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def _linear(self, args, copy):
        out_size = copy * self._num_units
        proj_size = args.get_shape()[-1]
        weights = vs.get_variable("kernel", [proj_size, out_size])
        out = math_ops.matmul(args, weights)
        if not self._layer_norm:
            bias = vs.get_variable("bias", [out_size])
            out = nn_ops.bias_add(out, bias)
        return out

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        with vs.variable_scope("gates"):
            h = state
            args = array_ops.concat([inputs, h], 1)
            concat = self._linear(args, 2)

            z, r = array_ops.split(value=concat, num_or_size_splits=2, axis=1)
            if self._layer_norm:
                z = self._norm(z, "update")
                r = self._norm(r, "reset")

        with vs.variable_scope("candidate"):
            args = array_ops.concat([inputs, math_ops.sigmoid(r) * h], 1)
            new_c = self._linear(args, 1)
            if self._layer_norm:
                new_c = self._norm(new_c, "state")
        new_h = self._activation(new_c) * math_ops.sigmoid(z) + \
                (1 - math_ops.sigmoid(z)) * h
        return new_h, new_h
