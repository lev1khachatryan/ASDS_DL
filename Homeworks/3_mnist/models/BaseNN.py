import tensorflow as tf
from data_loader import *
from utils import *
from abc import abstractmethod
import numpy as np
import pandas as pd
import keras.preprocessing.image
# import sklearn.preprocessing
import os;
import datetime  
import cv2

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)

        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.height_of_image = height_of_image
        self.width_of_image = width_of_image
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.base_dir = base_dir
        self.max_to_keep = max_to_keep
        self.model_name = model_name

        ####
        self.keep_prob = 0.33 # keeping probability with dropout regularization 
        self.index_in_epoch = 0
        self.current_epoch = 0
        self.n_log_step = 0 # counting current number of mini batches trained on
        
        # permutation array
        self.perm_array = np.array([])
        ####
        
    # function to get the next mini batch
    def next_mini_batch(self, mb_size):
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
            
            if self.train_on_augmented_data:
                # use augmented data for the next epoch
                self.x_train_aug = normalize_data(self.generate_images(self.x_train))
                self.y_train_aug = self.y_train
                
        end = self.index_in_epoch
        
        if self.train_on_augmented_data:
            # use augmented data
            x_tr = self.x_train_aug[self.perm_array[start:end]]
            y_tr = self.y_train_aug[self.perm_array[start:end]]
        else:
            # use original data
            x_tr = self.x_train[self.perm_array[start:end]]
            y_tr = self.y_train[self.perm_array[start:end]]
        
        return x_tr, y_tr
    
    # generate new images via rotations, translations, zoom using keras
    def generate_images(self, imgs):
    
        print('generate new set of images')
        
        # rotations, translations, zoom
        image_generator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 10, width_shift_range = 0.1 , height_shift_range = 0.1,
            zoom_range = 0.1)

        # get transformed images
        imgs = image_generator.flow(imgs.copy(), np.zeros(len(imgs)),
                                    batch_size=len(imgs), shuffle = False).next()    

        return imgs[0]

    # attach summaries to a tensor for TensorBoard visualization
    def summary_variable(self, var, var_name):
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    # function to create the network
    def create_network(self):
        tf.reset_default_graph()

        # variables for input and output 
        self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.height_of_image, self.width_of_image, self.num_channels], name='x_data_tf')
        self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='y_data_tf')

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
    
    def attach_summary(self, sess):
        
        # create summary tensors for tensorboard
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
        filepath = os.path.join(os.getcwd(), self.base_dir, (self.model_name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

    # helper function to train the model
    def train_model_helper(self, sess, x_train, y_train, x_valid, y_valid, n_epoch = 1, 
                    train_on_augmented_data = False):        
        # train on original or augmented data
        self.train_on_augmented_data = train_on_augmented_data
        
        # use augmented data
        if self.train_on_augmented_data:
            print('generate new set of images')
            self.x_train_aug = normalize_data(self.generate_images(self.x_train))
            self.y_train_aug = self.y_train
        
        # parameters
        mb_per_epoch = self.x_train.shape[0]/self.train_batch_size
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        # start timer
        start = datetime.datetime.now();
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ',self.learning_rate,', n_epoch = ', n_epoch,
              ', mb_size = ', self.train_batch_size)
        # looping over mini batches
        for i in range(int(n_epoch*mb_per_epoch)+1):            
            # get new batch
            x_batch, y_batch = self.next_mini_batch(self.train_batch_size)

            # run the graph
            sess.run(self.train_step_tf, feed_dict={self.x_data_tf: x_batch, 
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
                
            # save tensors and summaries of model
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
        self.display_step = display_step
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.summary_step = summary_step
        
        # training and validation data
        self.x_train, self.y_train = self.data_loader.all_train_data_loader()
        self.x_valid, self.y_valid = self.data_loader.all_val_data_loader()

        self.x_train = self.x_train.reshape(-1, self.height_of_image, self.width_of_image, self.num_channels)
        self.x_valid = self.x_valid.reshape(-1, self.height_of_image, self.width_of_image, self.num_channels)

        self.saver_tf = tf.train.Saver()
        # start timer
        start = datetime.datetime.now();

        # start tensorflow session
        with tf.Session() as sess:

            # attach summaries
            self.attach_summary(sess)

            # variable initialization of the default graph
            sess.run(tf.global_variables_initializer()) 

            # training on original data
            self.train_model_helper(sess, self.x_train, self.y_train, self.x_valid, self.y_valid, n_epoch = self.num_epochs)

            # training on augmented data
            # self.train_model_helper(sess, x_train, y_train, x_valid, y_valid, n_epoch = 14.0,
            #                     train_on_augmented_data = True)

            # save tensors and summaries of model
            self.save_model(sess)

        print('total running time for training: ', datetime.datetime.now() - start)

    # save tensors/summaries
    def save_model(self, sess):
        # tf saver
        filepath = os.path.join(os.getcwd(), self.model_name)
        self.saver_tf.save(sess, filepath)
        # tb summary
        self.train_writer.close()
        self.valid_writer.close()
        
        return None
  
    # forward prediction of current graph
    def forward(self, sess, x_data):
        y_pred_proba = self.y_pred_proba_tf.eval(session = sess, 
                                                 feed_dict = {self.x_data_tf: x_data,
                                                              self.keep_prob_tf: 1.0})
        return y_pred_proba
    
    # load session from file, restore graph, and load tensors
    def load_session_from_file(self, filename):
        tf.reset_default_graph()
        filepath = os.path.join(os.getcwd(), filename + '.meta')
        saver = tf.train.import_meta_graph(filepath)
        print(filepath)
        sess = tf.Session()
        saver.restore(sess, self.model_name)
        graph = tf.get_default_graph()
        self.load_tensors(graph)
        return sess
    
    def test_model(self):
        x_test, y_test = self.data_loader.all_test_data_loader()
        x_test = x_test.reshape(-1, self.height_of_image, self.width_of_image, self.num_channels)
        
        sess = self.load_session_from_file(self.model_name) # receive session 
        y_test_pred = {}
        y_test_pred_labels = {}
        y_test_pred[self.model_name] = self.forward(sess, x_test)
        sess.close()
        y_test_pred_labels[self.model_name] = one_hot_to_dense(y_test_pred[self.model_name])
        y_test = one_hot_to_dense(y_test)
        
        print('Test Accuracy: ', self.metrics(y_test, y_test_pred_labels[self.model_name]))
        return self.metrics(y_test, y_test_pred_labels[self.model_name])
        
    def initialize_network(self):
    	filepath = os.path.join(os.getcwd(), self.model_name + '.meta')
    	if os.path.isdir(filepath):
    		self.load_session_from_file(self.model_name)
    	return None
    
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')