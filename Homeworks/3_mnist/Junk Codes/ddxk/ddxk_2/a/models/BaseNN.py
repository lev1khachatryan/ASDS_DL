import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import random # to shuffle
import math

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.height_of_image = height_of_image
        self.width_of_image = width_of_image 
        self.num_classes= num_classes
        self.learning_rate = learning_rate 
        self.train_batch_size = train_batch_size
        self.num_epochs = num_epochs
        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir
        self.val_batch_size= val_batch_size
        self.test_batch_size = test_batch_size
        self.test_images_dir =test_images_dir
        self.base_dir = base_dir
        self.max_to_keep = max_to_keep
        self.model_name = model_name

    def create_network(self):
        self.X = tf.placeholder('float', [None, self.height_of_image*self.width_of_image], name = 'X')
        self.Y = tf.placeholder('float', [None, self.num_classes], name = 'Y')
        self.Y_pred = self.network(self.X)
        self.cost = self.metrics(self.Y, self.Y_pred)[0]
        self.accuracy = self.metrics(self.Y, self.Y_pred)[1]
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
        # return X, Y, opt, cost, accuracy, Y_pred
      
    def initialize_network(self):
        self.sess = tf.InteractiveSession()
        if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
          self.sess.run(tf.global_variables_initializer())
        else:
          new_saver = tf.train.import_meta_graph('my-model.meta')
          new_saver.restore(self.sess, tf.train.latest_checkpoint('./'))         
        return self.sess       
      
    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):     
            epoch_cost = 0       
            val_data_X = []
            val_data_Y = []    
            epoch_accuracy = 0
            for image in range(0, math.ceil(len(self.val_paths)/self.val_batch_size)):
                val_X, val_Y = self.data_loader.train_data_loader(image)
                val_data_X = val_data_X + val_X
                val_data_Y = val_data_Y + val_Y
                
            # X, Y, opt, cost, accuracy, Y_pred = self.create_network()
            accuracy_ = self.metrics(self.Y, self.Y_pred)[1]
            
            # self.initialize_network() #START!

            for epoch in range(self.num_epochs):            
                for i in range(math.ceil(len(self.train_paths)/ self.train_batch_size)):
                    minibatch_X, minibatch_Y = self.data_loader.train_data_loader(i)
                    _, mini_cost_b, mini_accuracy_b, mini_Y_pred_b = self.sess.run([self.opt, self.cost, self.accuracy, self.Y_pred], feed_dict = {self.X: minibatch_X, self.Y: minibatch_Y})
                    epoch_cost += mini_cost_b
                    epoch_accuracy += mini_accuracy_b
                    # print('Epoch: ', epoch, i, epoch_cost)
                epoch_cost = epoch_cost/math.ceil(len(self.train_paths)/self.train_batch_size)
                epoch_accuracy = epoch_accuracy/math.ceil(len(self.train_paths)/self.train_batch_size)
                
                if epoch%validation_step == 0:
                    accuracy_.eval({self.X: val_data_X, self.Y: val_data_Y})
                    
                if epoch%checkpoint_step == 0:
                    saver = tf.train.Saver(max_to_keep = self.max_to_keep)
                    if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                      os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                      saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                    else:
                      saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints','my-model'))
                      
                if epoch%display_step == 0:
                  print('Cost after ' + str(epoch) + 'th epoch is: '+ str(epoch_cost))
                  print('Train accuracy is: ' + str(epoch_accuracy))
                  
                if epoch%summary_step == 0:
                    if os.path.isfile(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
                        os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
                        tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                    else:
                        tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)    
            self.sess.close() 
                        
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
      