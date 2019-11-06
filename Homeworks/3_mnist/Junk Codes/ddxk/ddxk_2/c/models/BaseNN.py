import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import random
import math

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)
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
    	x = tf.placeholder("float", [None, self.height_of_image*self.width_of_image], name = "x")
    	y = tf.placeholder("float", [None, self.num_classes], name="y")
    	prediction = self.network(x)
    	loss, accuracy = self.metrics(y, prediction)
    	optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
    	return x, y, optimizer, loss, accuracy, prediction

    def initialize_network(self):
    	self.sess = tf.InteractiveSession()
    	if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
    		self.sess.run(tf.global_variables_initializer())

    	else:
    		new_saver = tf.train.import_meta_graph('my-model.meta')
    		new_saver.restore(self.sess, tf.train.latest_checkpoint('./'))
    	return self.sess

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        val_data = []
        epoch_loss = 0
        epoch_accuracy = 0
        #x_val,y_val=self.data_loader.val_data_loader(0)
        #x_test,y_test=self.data_loader.test_data_loader(0)
        x_val=[]
        y_val=[]
       	for i in range(0, math.ceil(len(self.val_paths)/ self.val_batch_size)):
       		x_val1,y_val1=self.data_loader.train_data_loader(i)
       		x_val=x_val+x_val1
       		y_val=y_val+y_val1
   
        x,y, optimizer, loss, accuracy, prediction=self.create_network()
        accuracy_m = self.metrics(y,prediction)[1]
        self.initialize_network()
        for i in range(0,self.num_epochs):
            random.shuffle(self.train_paths)
            for j in range(0, math.ceil(len(self.train_paths)/ self.train_batch_size)):
                minibatch_x, minibatch_y = self.data_loader.train_data_loader(j)
                _, loss_b, accuracy_b, prediction_b  = self.sess.run([optimizer, loss, accuracy, prediction], feed_dict = {x: minibatch_x, y: minibatch_y})
                epoch_loss += loss_b
                epoch_accuracy += accuracy_b

            epoch_loss = epoch_loss/math.ceil(len(self.train_paths)/ self.train_batch_size)
            epoch_accuracy = epoch_accuracy/math.ceil(len(self.train_paths)/ self.train_batch_size)

            if i%validation_step == 0:
                accuracy_m.eval({x: x_val, y: y_val})

            if i%checkpoint_step == 0: 
                saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                    saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                   
               	else:
               		saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints','my-model'))


            if i%display_step == 0:
                print("cost after epoch %i :  %.3f" % (i + 1, epoch_loss), end="")
                print("train accuracy   :  %.3f" % epoch_accuracy)

            if i%summary_step == 0:
            	if os.path.isfile(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
            		os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
            		tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
            	else:
            		tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
				

					

    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
