import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from utils import clear_start, next_batch, next_batch_shuffle, load_graph, freeze_save_graph
from os import path
import time


def length(a):
    return tf.sqrt(tf.reduce_sum(tf.square(a), axis=1))


def sim_matrix(a):
    return tf.tensordot(a, tf.transpose(a), 1) / tf.multiply(length(a), length(a))


class AutoEncoder:
    """AutoEncoder
    Parameters
    ------------
    structure : list (default=[10, 5, 10])
        number of neurons in each layer (including input and output layers)
    encoding_layer_index: int (default=1)
        which layer represents encoding (0-indexed)
    activation_fn: function (default tf.nn.relu)
        activation function for hidden layers
    verbose: bool (default=False)
        enable verbose output
    cpu_only: bool (default=True)
        use only cpu
    cpu_number: int (default=0)
        which cpu to use
    gpu_fraction : float (default=0.7)
        between (0.0-1.0) how much of the gpu memory allow to use, used if cpu_only is false
    gpu_number: int (default=0)
        which gpu to use
    log_dir : string (default='/tmp/log/')
        path to where save the logs of training (tensorboard directory)
    random_state : int
        set random state
    """

    def __init__(self, structure=[10, 5, 10],
                 encoding_layer_index=1,
                 activation_fn=tf.nn.relu,
                 verbose=False,
                 cpu_only=True, cpu_number=0,
                 gpu_fraction=0.7, gpu_number=0,
                 log_dir='/tmp/log/',
                 random_state=None):

        self.structure = structure
        self.encoding_layer_index = encoding_layer_index
        assert len(structure) > 2, 'the nerual network should have at least 3 layers: input, hidden, output'
        assert self.structure[0] == self.structure[-1], 'The input and output dimensions should match'
        assert ((0 < encoding_layer_index) and (encoding_layer_index < len(self.structure) - 1)), 'encoding layer ' \
                                                                                                  'cannot be input or' \
                                                                                                  ' output layer '
        if activation_fn is None:
            self.activation_fn = tf.identity
        else:
            self.activation_fn = activation_fn
        self._ld = log_dir
        self._verbose = verbose
        assert self.structure[-1] > 1, 'you should have at least two classes'
        self._cpu_only = cpu_only

        if cpu_only:
            self._config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
            self._cpu_number = cpu_number
        else:
            self._gpu_number = gpu_number
            self._config = tf.ConfigProto(allow_soft_placement=True)
            self._config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self._random_state = None or random_state
        self._input_ph = None
        self._dropout_keep_rate = None
        self._logits = None
        self._output = None
        self._labels = None
        self._sess = None
        self._network = None
        self.total_losses = None
        self.best_loss = None
        self._encoding = None
        self._loss1 = None
        self._loss2 = None
        self._loss = None
        self._correct_prediction = None
        self._evaluation_step = None
        self._num_batches_train = None
        self._summary_op_step = None
        self._batch_step = None
        self._batch_size_val = None
        self._train_vars = None
        self._summary_op = None
        self._train_op = None
        self._start = None
        self._epoch = None

    def define_loss(self, alpha, beta):
        # sim_input = sim_matrix(self._input_ph)
        # sim_encoded = sim_matrix(self._encoding)

        try:
            tf.summary.histogram('encoding', self._encoding)
            tf.summary.histogram('reconstructed', self._output)
            tf.summary.histogram('inputs', self._input_ph)
        except BaseException as e:
            if self._verbose:
                print(str(e))

        with tf.name_scope('losses'):
            self._loss1 = tf.multiply(tf.reduce_mean(tf.square(tf.subtract(self._input_ph, self._output))),
                                      alpha, name='dist')
            self._loss2 = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in self._train_vars]),
                                      beta, name='l2_reg_loss')
            # self._loss3 = tf.reduce_mean(tf.square(tf.subtract(sim_input, sim_encoded)), name='similarity')
            # self._loss = tf.add_n((self._loss1, self._loss2, self._loss3), name='total_loss')
            self._loss = tf.add_n((self._loss1, self._loss2), name='total_loss')
            tf.summary.scalar('distance', self._loss1)
            tf.summary.scalar('l2_loss', self._loss2)
            # tf.summary.scalar('similarity', self._loss3)
            tf.summary.scalar('total_loss', self._loss)

    def _construct_nn(self, use_batch_norm, seperate_validation):
        tf.reset_default_graph()
        clear_start([self._ld])
        if self._random_state is not None:
            if self._verbose:
                print('seed is fixed to {}'.format(self._random_state))
            tf.set_random_seed(self._random_state)
            np.random.seed(self._random_state)
        layers = []

        self._input_ph = tf.placeholder(tf.float32, shape=[None, self.structure[0]], name='input')
        self._dropout_keep_rate = tf.placeholder_with_default(1., shape=None, name='keep_rate')
        self._train_mode = tf.placeholder_with_default(False, shape=None, name='train_mode')
        layers.append(self._input_ph)
        j = 1
        with tf.variable_scope('autoencoder'):
            for i, n_neurons in enumerate(self.structure[1:-1]):

                if j == 1:
                    x = tf.layers.dense(self._input_ph, n_neurons, name='hidden_%s' % j,
                                        kernel_initializer=tf.truncated_normal_initializer())
                else:
                    x = tf.layers.dense(x, n_neurons, name='hidden_%s' % j,
                                        kernel_initializer=tf.truncated_normal_initializer())
                if use_batch_norm:
                    x = tf.layers.batch_normalization(x, axis=1, training=self._train_mode, scale=False)
                    layers.append(x)
                x = self.activation_fn(x)
                layers.append(x)
                x = tf.layers.dropout(x, tf.subtract(1., self._dropout_keep_rate), name='dropout_%s' % j)
                layers.append(x)
                if j == self.encoding_layer_index:
                    x = tf.identity(x, name='encoding')
                    self._encoding = x
                j += 1
        self._output = tf.layers.dense(x, self.structure[-1], name='output',
                                       kernel_initializer=tf.truncated_normal_initializer())
        self._labels = tf.placeholder(tf.float32, shape=[None, self.structure[-1]], name='label')
        layers.append(self._output)
        if self._cpu_only:
            with tf.device('/cpu:{}'.format(self._cpu_number)):
                sess = tf.Session(config=self._config)
                if seperate_validation:
                    self._train_writer = tf.summary.FileWriter(self._ld + 'train/', sess.graph)
                    self._val_writer = tf.summary.FileWriter(self._ld + 'val/', sess.graph)
                else:
                    self._train_writer = tf.summary.FileWriter(self._ld, sess.graph)
        else:
            with tf.device('/gpu:{}'.format(self._gpu_number)):
                sess = tf.Session(config=self._config)
                if seperate_validation:
                    self._train_writer = tf.summary.FileWriter(self._ld + 'train/', sess.graph)
                    self._val_writer = tf.summary.FileWriter(self._ld + 'val/')
                else:
                    self._train_writer = tf.summary.FileWriter(self._ld, sess.graph)
        self._sess = sess
        self._network = layers

    def fit(self, X,
            seperate_validation=True, validation_ratio=0.2,
            learning_rate=0.01, alpha=1., beta=0.0005,
            n_epochs=10, batch_size=16,
            use_batch_norm=True,
            batch_norm_train=True,
            dropout_keep_rate=1.,
            early_stopping_epochs=None,
            early_stopping_method='dlr',
            early_stopping_iters=2,
            save_best_model=False,
            continue_fit=False):

        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param y: array-like, shape (n_samples,)
        Target vector relative to X.
        :param loss_type: string, 'rmse' or 'crossentropy'
        :param seperate_validation: bool, seperate validation set from X
        :param validation_ratio: float, between (0.0-1.0) the ratio of seperated validation (default=0.2)
        :param learning_rate: float, the learning rate
        :param beta: float, L2 regularization parameter for the weights
        :param batch_norm_train: bool, train batch_normalization parameters
        :param n_epochs: int, number of epochs to train the network
        :param batch_size: int, batch size
        :param dropout_keep_rate: float, 0.6 would drop 40% of weights
        :param early_stopping_epochs: int, how many epochs to train without improvement (default=None)
        :param early_stopping_method: string, 'dlr' for multiplying learning rate by 0.1 or 'stop'
        :param early_stopping_iters: int, how many time to decrease learning rate (used if 'dlr' is True)
        :param save_best_model: bool, whether or not to save the model with lowest loss
        :param continue_fit, bool, do not reset the graph and continue with current weights
        :return:
        """

        X = np.array(X, np.float32)
        if not continue_fit:
            self._construct_nn(use_batch_norm, seperate_validation)

        if seperate_validation:
            X, val_X = train_test_split(X, test_size=validation_ratio, random_state=self._random_state)
            assert len(X) > 0, "The training set is empty"
            assert len(val_X) > 0, "The validation set is empty"

        if batch_size is None:
            batch_size = len(X)
        train_inds = np.arange(len(X))
        if not continue_fit:
            self._num_batches_train = len(train_inds) // batch_size
            b_w = 1. / self._num_batches_train
            self._summary_op_step = int(pow(10, np.ceil(np.log10(self._num_batches_train))))
            self._batch_step = int(np.floor(self._summary_op_step * b_w))

        if seperate_validation:
            val_inds = np.arange(len(val_X))
            self._batch_size_val = len(val_inds) // self._num_batches_train

        if not continue_fit:
            train_vars = tf.trainable_variables()
            l2_optimizable_vars = [i for i in train_vars if 'kernel' in i.name.split('/')[-1]]
            self._train_vars = l2_optimizable_vars

            with tf.name_scope('losses'):
                self.define_loss(alpha=alpha, beta=beta)

            self._summary_op = tf.summary.merge_all()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)
            self._sess.run(tf.global_variables_initializer())

        if not continue_fit:
            self._start = time.time()
            self.total_losses = []
            self.best_loss = np.inf

        sleep_time = 0.2
        epochs_not_improved = 0
        decreased_learning_rate = 0

        for epoch in range(n_epochs):
            if continue_fit:
                self._epoch += 1
            else:
                self._epoch = epoch
            # batch_accuracies = []
            if self._verbose:
                print('epoch %d started' % (self._epoch))

            q = 0
            cummulative_loss = 0
            # for j in tqdm(range(self._num_batches_train)):
            for j in range(self._num_batches_train):
                train_inds, batch_inds = next_batch_shuffle(train_inds, batch_size)
                batch_features = X[batch_inds]

                _, train_summary, _loss = self._sess.run([self._train_op, self._summary_op, self._loss],
                                                         feed_dict={self._input_ph: batch_features,
                                                                    self._labels: batch_features,
                                                                    self._dropout_keep_rate: dropout_keep_rate,
                                                                    self._train_mode: batch_norm_train})

                if seperate_validation:
                    val_inds, batch_inds = next_batch(val_inds, self._batch_size_val)
                    batch_features = val_X[batch_inds]
                    assert len(batch_features) > 0, 'empty batch while validation'
                    val_summary, _loss = self._sess.run([self._summary_op, self._loss1],
                                                        feed_dict={self._input_ph: batch_features,
                                                                   self._labels: batch_features})

                    self._val_writer.add_summary(val_summary,
                                                 self._epoch * self._summary_op_step + j * self._batch_step)

                cummulative_loss += _loss
                q += 1

                self._train_writer.add_summary(train_summary,
                                               self._epoch * self._summary_op_step + j * self._batch_step)

            mean_loss = cummulative_loss / q
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                if save_best_model:
                    self.save_pb(self._ld + 'best.pb')
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1

            if self._verbose:
                print('%d epoch mean loss: %f' % (self._epoch, mean_loss))
            self.total_losses.append(mean_loss)

            if early_stopping_epochs is not None:
                if epochs_not_improved > early_stopping_epochs:
                    if early_stopping_method == 'stop':
                        if self._verbose:
                            print('early stopping')
                        break
                    elif early_stopping_method == 'dlr':
                        if decreased_learning_rate > early_stopping_iters:
                            if self._verbose:
                                print('early stopping')
                            break
                        learning_rate /= 10
                        decreased_learning_rate += 1
                        epochs_not_improved = 0
                        if self._verbose:
                            print('new learning rate {}'.format(learning_rate))
                    else:
                        if self._verbose:
                            print('unknown method for early stopping. stopping the training.')
                        break

            train_inds = np.arange(len(X))
            if seperate_validation:
                val_inds = np.arange(len(val_X))
            time.sleep(sleep_time)
        if self._verbose:
            print('The training took {} seconds'.format(time.time() - self._start - self._epoch * sleep_time))

    def fit_transform(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param batch_size: int, batch size
        :return: predicted encodings
        """
        X = np.array(X)
        inds = np.arange(len(X))
        predictions = []
        batch_size = batch_size or len(X)

        start = time.time()
        while len(inds) > 0:
            inds, batch_inds = next_batch(inds, batch_size)

            batch_features = X[batch_inds]
            batch_preds = self._sess.run(self._encoding, feed_dict={self._input_ph: batch_features})
            predictions.extend(batch_preds)

        if self._verbose:
            print('The inference took {} seconds'.format(time.time() - start))
        predictions = np.squeeze(predictions)
        return predictions

    def score(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param y: array-like, shape (n_samples,)
        Target vector relative to X.
        :param batch_size: int, batch size
        :return: mean l2 distance between input and output
        """
        X = np.array(X)
        inds = np.arange(len(X))
        predictions = []
        batch_size = batch_size or len(X)
        while len(inds) > 0:
            inds, batch_inds = next_batch(inds, batch_size)

            batch_features = X[batch_inds]
            batch_preds = self._sess.run(self._output, feed_dict={self._input_ph: batch_features})
            predictions.extend(batch_preds)
        predictions = np.array(predictions)

        return np.linalg.norm(X - predictions, axis=-1).mean()

    def predict(self, X, batch_size=None):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        :param batch_size: int, batch size
        :return: decoded predicted encodings
        """
        X = np.array(X)
        inds = np.arange(len(X))
        predictions = []
        batch_size = batch_size or len(X)

        start = time.time()
        while len(inds) > 0:
            inds, batch_inds = next_batch(inds, batch_size)

            batch_features = X[batch_inds]
            batch_preds = self._sess.run(self._output, feed_dict={self._input_ph: batch_features})
            predictions.extend(batch_preds)

        if self._verbose:
            print('The inference took {} seconds'.format(time.time() - start))
        predictions = np.squeeze(predictions)
        return predictions

    def save_pb(self, path_to_pb):
        freeze_save_graph(self._sess, path.basename(path_to_pb), 'output/BiasAdd', path.dirname(path_to_pb))

    def load_pb(self, path_to_pb):
        if self._cpu_only:
            with tf.device('/cpu:{}'.format(self._cpu_number)):
                self._input_ph, self._encoding, self._output = load_graph(
                    path_to_pb, ['input:0', 'autoencoder/encoding:0', 'output/BiasAdd:0'])
                self._sess = tf.Session(config=self._config)
        else:
            with tf.device('/gpu:{}'.format(self._gpu_number)):
                self._input_ph, self._encoding, self._output = load_graph(
                    path_to_pb, ['input:0', 'autoencoder/encoding:0', 'output/BiasAdd:0'])
                self._sess = tf.Session(config=self._config)

        return self
