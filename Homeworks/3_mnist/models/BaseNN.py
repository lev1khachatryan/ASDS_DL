import tensorflow as tf
from data_loader import *
from utils import *
from abc import abstractmethod
import numpy as np
import pandas as pd
import os;
import datetime  
import cv2

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.base_dir = base_dir
        self.max_to_keep = max_to_keep
        self.model_name = model_name
        self.keep_prob = keep_prob

        ####
        self.index_in_epoch = 0
        self.current_epoch = 0
        self.n_log_step = 0 # counting current number of mini batches trained on

        # permutation array
        self.perm_array = np.array([])
        ####
    
    def create_network(self):
        """
        Create base components of the network.
        Main structure of network will be described in network function.
        -----------------
        Parameters:
            None
        Returns:
            None
        -----------------
        """
        tf.reset_default_graph()

        # variables for input and output 
        self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.height_of_image, self.data_loader.width_of_image, self.data_loader.num_channels], name='x_data_tf')
        self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.data_loader.num_classes], name='y_data_tf')

        self.z_pred_tf = self.network(self.x_data_tf)

        # cost function
        self.cross_entropy_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_data_tf, logits=self.z_pred_tf), name = 'cross_entropy_tf')

        # optimisation function
        self.learn_rate_tf = tf.placeholder(dtype=tf.float32, name="learn_rate_tf")
        self.train_step_tf = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
            self.cross_entropy_tf, name = 'train_step_tf')

        # predicted probabilities in one-hot encoding
        self.y_pred_proba_tf = tf.nn.softmax(self.z_pred_tf, name='y_pred_proba_tf') 
        
        # tensor of correct predictions
        self.y_pred_correct_tf = tf.equal(tf.argmax(self.y_pred_proba_tf, 1),
                                          tf.argmax(self.y_data_tf, 1),
                                          name = 'y_pred_correct_tf')  
        
        # accuracy 
        self.accuracy_tf = tf.reduce_mean(tf.cast(self.y_pred_correct_tf, dtype=tf.float32),
                                         name = 'accuracy_tf')

        # tensors to save intermediate accuracies and losses during training
        self.train_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                         name='train_loss_tf', validate_shape = False)
        self.valid_loss_tf = tf.Variable(np.array([]), dtype=tf.float32,
                                         name='valid_loss_tf', validate_shape = False)
        self.train_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                        name='train_acc_tf', validate_shape = False)
        self.valid_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, 
                                        name='valid_acc_tf', validate_shape = False)

        return None

    def summary_variable(self, var, var_name):
        """
        Attach summaries to a tensor for TensorBoard visualization
        -----------------
        Parameters:
            var         - variable we want to attach
            var_name    - name of the variable
        Returns:
            None
        -----------------
        """
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        return None
    
    def attach_summary(self, sess):
        """
        Create summary tensors for tensorboard.
        -----------------
        Parameters:
            sess - the session for which we want to create summaries
        Returns:
            None
        -----------------
        """
        self.summary_variable(self.W_conv1_tf, 'W_conv1_tf')
        self.summary_variable(self.b_conv1_tf, 'b_conv1_tf')
        self.summary_variable(self.W_conv2_tf, 'W_conv2_tf')
        self.summary_variable(self.b_conv2_tf, 'b_conv2_tf')
        self.summary_variable(self.W_conv3_tf, 'W_conv3_tf')
        self.summary_variable(self.b_conv3_tf, 'b_conv3_tf')
        self.summary_variable(self.W_fc1_tf, 'W_fc1_tf')
        self.summary_variable(self.b_fc1_tf, 'b_fc1_tf')
        self.summary_variable(self.W_fc2_tf, 'W_fc2_tf')
        self.summary_variable(self.b_fc2_tf, 'b_fc2_tf')
        tf.summary.scalar('cross_entropy_tf', self.cross_entropy_tf)
        tf.summary.scalar('accuracy_tf', self.accuracy_tf)

        # merge all summaries for tensorboard
        self.merged = tf.summary.merge_all()

        # initialize summary writer 
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'logs', (self.model_name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

        return None

    def close_writers(self):
        """
        Close train and validation summary writer.
        -----------------
        Parameters:
            sess - the session we want to save
        Returns:
            None
        -----------------
        """
        self.train_writer.close()
        self.valid_writer.close()

        return None

    def save_model(self, sess):
        """
        Save tensors/summaries
        -----------------
        Parameters:
            sess - the session we want to save
        Returns:
            None
        -----------------
        """
        filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'checkpoints', self.model_name)
        self.saver_tf.save(sess, filepath)
        
        return None

    # function to get the next mini batch
    def next_mini_batch(self, mb_size): ## Will be deleted in the future
        start = self.index_in_epoch
        self.index_in_epoch += mb_size
        self.current_epoch += mb_size/len(self.x_train)  
        
        # adapt length of permutation array
        if not len(self.perm_array) == len(self.x_train):
            self.perm_array = np.arange(len(self.x_train))
        
        # shuffle once at the start of epoch
        if start == 0:
            np.random.shuffle(self.perm_array)

        # at the end of the epoch
        if self.index_in_epoch > self.x_train.shape[0]:
            np.random.shuffle(self.perm_array) # shuffle data
            start = 0 # start next epoch
            self.index_in_epoch = mb_size # set index to mini batch size
                
        end = self.index_in_epoch

        x_tr = self.x_train[self.perm_array[start:end]]
        y_tr = self.y_train[self.perm_array[start:end]]

        return x_tr, y_tr

    def train_model_helper(self, sess, x_train, y_train, x_valid, y_valid, n_epoch = 1):        
        """
        Helper function to train the model.
        -----------------
        Parameters:
            sess - the session for which we want to create summaries
            x_train (matrix_like) - train images
            y_train (matrix_like) - labels of train images
            x_valid (matrix_like) - validation images
            y_valid (matrix_like) - labels of validation images
            n_epoch (int)         - number of epochs
        Returns:
            None
        -----------------
        """
        
        # parameters
        mb_per_epoch = self.x_train.shape[0]/self.data_loader.train_batch_size
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        # start timer
        start = datetime.datetime.now();
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ', self.learning_rate,', n_epoch = ', n_epoch,
              ', mb_size = ', self.data_loader.train_batch_size)
        # looping over mini batches
        for i in range(int(n_epoch*mb_per_epoch)+1):            
            # get new batch
            x_batch, y_batch = self.next_mini_batch(self.data_loader.train_batch_size)

            # run the graph
            self.sess.run(self.train_step_tf, feed_dict={self.x_data_tf: x_batch, 
                                                    self.y_data_tf: y_batch, 
                                                    self.keep_prob_tf: self.keep_prob, 
                                                    self.learn_rate_tf: self.learning_rate})
            
            feed_dict_valid = {self.x_data_tf: self.x_valid, 
                               self.y_data_tf: self.y_valid, 
                               self.keep_prob_tf: 1.0}
            # feed_dict_train = {self.x_data_tf: self.x_train[self.perm_array[:len(self.x_valid)]], 
            #                     self.y_data_tf: self.y_train[self.perm_array[:len(self.y_valid)]], 
            #                     self.keep_prob_tf: 1.0}
            feed_dict_train = {self.x_data_tf: x_batch, 
                                self.y_data_tf: y_batch, 
                                self.keep_prob_tf: 1.0}
            
            # store losses and accuracies
            if i%self.validation_step == 0:
                valid_loss.append(sess.run(self.cross_entropy_tf,
                                           feed_dict = feed_dict_valid))
                valid_acc.append(self.accuracy_tf.eval(session = sess, 
                                                       feed_dict = feed_dict_valid))
                print('%.2f epoch, %.2f iteration: val loss = %.4f, val acc = %.4f'%(
                    self.current_epoch, i, valid_loss[-1],valid_acc[-1]))

            # summary for tensorboard
            if i%self.summary_step == 0:
                self.n_log_step += 1 # for logging the results
                train_summary = sess.run(self.merged, feed_dict={self.x_data_tf: x_batch, 
                                                                self.y_data_tf: y_batch, 
                                                                self.keep_prob_tf: 1.0})
                valid_summary = sess.run(self.merged, feed_dict = feed_dict_valid)
                self.train_writer.add_summary(train_summary, self.n_log_step)
                self.valid_writer.add_summary(valid_summary, self.n_log_step)
                
            if i%self.display_step == 0:
                train_loss.append(sess.run(self.cross_entropy_tf,
                                           feed_dict = feed_dict_train))
                train_acc.append(self.accuracy_tf.eval(session = sess, 
                                                       feed_dict = feed_dict_train))
                print('%.2f epoch, %.2f iteration: train loss = %.4f, train acc = %.4f'%(
                    self.current_epoch, i,  train_loss[-1],train_acc[-1]))
                
            # save current model to disk
            if i%self.checkpoint_step == 0:
                self.save_model(sess)
                
        # concatenate losses and accuracies and assign to tensor variables
        tl_c = np.concatenate([self.train_loss_tf.eval(session=sess), train_loss], axis = 0)
        vl_c = np.concatenate([self.valid_loss_tf.eval(session=sess), valid_loss], axis = 0)
        ta_c = np.concatenate([self.train_acc_tf.eval(session=sess), train_acc], axis = 0)
        va_c = np.concatenate([self.valid_acc_tf.eval(session=sess), valid_acc], axis = 0)
   
        sess.run(tf.assign(self.train_loss_tf, tl_c, validate_shape = False))
        sess.run(tf.assign(self.valid_loss_tf, vl_c , validate_shape = False))
        sess.run(tf.assign(self.train_acc_tf, ta_c , validate_shape = False))
        sess.run(tf.assign(self.valid_acc_tf, va_c , validate_shape = False))
        
        print('running time for training: ', datetime.datetime.now() - start)
        return None
    
    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        """
        Main function for model training.
        -----------------
        Parameters:
            display_step       (int)   - Number of steps we cycle through before displaying detailed progress
            validation_step    (int)   - Number of steps we cycle through before validating the model
            checkpoint_step    (int)   - Number of steps we cycle through before saving checkpoint
            summary_step       (int)   - Number of steps we cycle through before saving summary
        Returns:
            None
        -----------------
        """
        self.display_step = display_step
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.summary_step = summary_step
        
        # training and validation data
        self.x_train, self.y_train = self.data_loader.all_train_data_loader()
        self.x_valid, self.y_valid = self.data_loader.all_val_data_loader()

        self.x_train = self.x_train.reshape(-1, self.data_loader.height_of_image, self.data_loader.width_of_image, self.data_loader.num_channels)
        self.x_valid = self.x_valid.reshape(-1, self.data_loader.height_of_image, self.data_loader.width_of_image, self.data_loader.num_channels)

        self.saver_tf = tf.train.Saver(max_to_keep = self.max_to_keep)

        # attach summaries
        self.attach_summary(self.sess)

        # variable initialization of the default graph
        self.sess.run(tf.global_variables_initializer()) 

        # training on original data
        self.train_model_helper(self.sess, self.x_train, self.y_train, self.x_valid, self.y_valid, n_epoch = self.num_epochs)

        # save final model
        self.save_model(self.sess)

        self.close_writers()

    def get_accuracy(self, sess):
        """
        Get accuracies of training and validation sets.
        -----------------
        Parameters:
            sess - session
        Returns:
            tuple (tuple of lists) train and validation accuracies
        -----------------
        """
        train_acc = self.train_acc_tf.eval(session = sess)
        valid_acc = self.valid_acc_tf.eval(session = sess)
        return train_acc, valid_acc

    def get_loss(self, sess):
        """
        Get losses of training and validation sets.
        -----------------
        Parameters:
            sess - session
        Returns:
            tuple (tuple of lists) train and validation losses
        -----------------
        """
        train_loss = self.train_loss_tf.eval(session = sess)
        valid_loss = self.valid_loss_tf.eval(session = sess)
        return train_loss, valid_loss 

    def forward(self, sess, x_data):
        """
        Forward prediction of current graph.
        Will be used in test_model method.
        -----------------
        Parameters:
            sess                 - actual session
            x_data (matrix_like) - data for which we want to calculate predicted probabilities
        Returns:
            vector_like - predicted probabilities for input data
        -----------------
        """
        y_pred_proba = self.y_pred_proba_tf.eval(session = sess, 
                                                 feed_dict = {self.x_data_tf: x_data,
                                                              self.keep_prob_tf: 1.0})
        return y_pred_proba
    
    def load_session_from_file(self, filename):
        """
        Load session from file, restore graph, and load tensors.
        -----------------
        Parameters:
            filename (string) - the name of the model (name of file we saved in disk)
        Returns:
            session
        -----------------
        """
        tf.reset_default_graph()

        filepath = os.path.join(os.getcwd(), filename + '.meta')
        # filepath = os.path.join(os.getcwd(), self.base_dir, filename + '.meta')
        # filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, filename + '.meta')
        # filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'checkpoints', filename + '.meta')
        print(filepath)
        saver = tf.train.import_meta_graph(filepath)
        sess = tf.Session()
        saver.restore(sess, self.model_name)
        graph = tf.get_default_graph()
        
        self.load_tensors(graph)
        
        return sess

    def test_model(self):
        """
        load model and test on test data.
        -----------------
        Parameters:
            None
        Returns:
            metric, defined in dnn class (for example accuracy)
        -----------------
        """
        x_test, y_test = self.data_loader.all_test_data_loader()
        x_test = x_test.reshape(-1, self.data_loader.height_of_image, self.data_loader.width_of_image, self.data_loader.num_channels)
        
        sess = self.load_session_from_file(self.model_name)
        
        y_test_pred = {}
        y_test_pred_labels = {}
        y_test_pred[self.model_name] = self.forward(sess, x_test)

        sess.close()
        
        y_test_pred_labels[self.model_name] = one_hot_to_dense(y_test_pred[self.model_name])
        y_test = one_hot_to_dense(y_test)
        
        print('Test Accuracy: ', self.metrics(y_test, y_test_pred_labels[self.model_name]))
        return self.metrics(y_test, y_test_pred_labels[self.model_name])
        
    # Initialize network from meta file
    # def initialize_network(self):
    # 	filepath = os.path.join(os.getcwd(), self.model_name + '.meta')
    # 	if os.path.isdir(filepath):
    # 		self.load_session_from_file(self.model_name)
    # 	return None

    def initialize_network(self):
        """
        Initialize network from last checkpoint if exists, otherwise initialize with random values.
        -----------------
        Parameters:
            None
        Returns:
            metric, defined in dnn class (for example accuracy)
        -----------------
        """
        self.sess = tf.InteractiveSession()
        filepath = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'checkpoints', self.model_name + '.meta')
        if ~os.path.isdir(filepath):
          self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = self.load_session_from_file(self.model_name)
        return None
    
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')