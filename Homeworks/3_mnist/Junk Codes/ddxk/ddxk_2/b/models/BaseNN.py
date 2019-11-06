import tensorflow as tf
from data_loader import *
from abc import abstractmethod
#from util import config
import math
import numpy as np
import random
from numpy import array

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)
        self.num_epochs=num_epochs
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)
        self.base_dir=base_dir

        self.train_batch_size=train_batch_size
        self. val_batch_size= val_batch_size
        self. test_batch_size= test_batch_size

        self.learning_rate=learning_rate
        self.width_of_image=width_of_image
        self.height_of_image=height_of_image
        self.num_classes=num_classes
        self.max_to_keep=max_to_keep
        self.model_name=model_name

    def create_network(self):
         # Creating place holders for image data and its labels
         ###########
        #X = tf.placeholder(tf.float32,[None, 784], name="X")
        #Y = tf.placeholder(tf.float32, [None, 10],name="Y")
        ############

        # x = tf.placeholder(tf.float32,[None,self.height_of_image,self.width_of_image] ,name="x")
        # y = tf.placeholder(tf.float32,[None,self.num_classes],  name="y")
        x = tf.placeholder(tf.float32,[None,self.height_of_image*self.width_of_image] ,name="x")
        y = tf.placeholder(tf.float32,[None,self.num_classes],  name="y")
        #y=np.reshape(y,(-1,1))

        # # m = tf.layers.dense(x, units=256, activation=tf.nn.relu,
        #                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                 bias_initializer=tf.contrib.layers.xavier_initializer())
        # m = tf.layers.dense(m, units=128, activation=tf.nn.relu,
        #                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                 bias_initializer=tf.contrib.layers.xavier_initializer())
        # prediction = tf.layers.dense(m, units=self.num_classes, activation=tf.nn.softmax, name="p", 
        #                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                 bias_initializer=tf.contrib.layers.xavier_initializer())
        prediction=self.network(x)

        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
        loss = self.metrics(y,prediction)[0]
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return x,y,prediction,loss,optim


        #self.input_images = tf.placeholder(tf.float32)
        #self.labels = tf.placeholder(tf.float32)
        #"compute" prediction of your network
        ############
        #initializer = tf.contrib.layers.xavier_initializer()
        #W1 = tf.Variable(initializer((784, 300)),name='W1')
        #b1 = tf.Variable(initializer([300]),name='b1')

        #layer1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
 
        #bW2 = tf.Variable(initializer((300,10)),name='W2')
        #b2 = tf.Variable(initializer([10]),name='b2')
        #layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,h2),b2))
 
        #W3 = tf.Variable(initializer((300,10)),name='W3')
        #b3 = tf.Variable(initializer([10]),name='b3')
 
        #layer3 = tf.nn.softmax(tf.add(tf.matmul(layer1,W2),b2))
 ##########################



        #loss function 
        #total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer3,labels = Y)
        #avg_loss = tf.reduce_mean(total_loss)
        #y_clipped = tf.clip_by_value(layer3, 1e-10, 0.9999999)
        #ncross_entropy = -tf.reduce_mean(tf.reduce_sum(Y* tf.log(y_clipped) + (1 - Y) * tf.log(1 - y_clipped), axis=1))
        #optimazer
        #optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        #optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(avg_loss)
        #avg_loss = compute_loss(layer3, Y)
        #optimizer = create_optimizer().minimize(avg_loss)
        

    def initialize_network(self):
        # self.sess= tf.Session() 
        self.sess= tf.InteractiveSession()
        #if  len(os.listdir('/base_dir/'))!= 0:
        if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
            #saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            # saver = tf.train.import_meta_graph('my-model.meta')

            # saver.restore(sess, tf.train.latest_checkpoint('./'))
            # new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        else:

            # sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('my-model.meta')

            saver.restore(self.sess, tf.train.latest_checkpoint('./'))
              # ete checkpointer ka  eta vercnum ete che random init ani 
        
        
        

        
       
        

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        num_complete_minibatches = math.ceil(len(self.train_paths)/ self.train_batch_size)
        x, y, pred,loss, optim=self.create_network()
        x_val,y_val=self.data_loader.val_data_loader(0)
        
        x_test,y_test=self.data_loader.test_data_loader(0)
        # y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        # y_val = np.asarray(y_val).astype('float32').reshape((-1,1))

        # x_val = []
        # y_val=[]
        # for s in range(math.ceil(len(self.val_paths)/ self.val_batch_size)):
        #     x_val+= self.data_loader.val_data_loader(s)[0]
        #     y_val+=self.data_loader.val_data_loader(s)[1]

        # x_test = []
        # y_test=[]
        # for s in range(math.ceil(len(self.test_paths)/ self.test_batch_size)):
        #     x_test+=self.data_loader.test_data_loader(s)[0]
        #     y_test+=self.data_loader.test_data_loader(s)[1]



        


        accuracy=self.metrics(y,pred)[1]
        # with tf.Session() as sess:
        self.initialize_network()
        
        # sess.run(init)  #initializes the variables created
        for epoch in range(self.num_epochs):
            random.shuffle(self.train_paths)
            # permutation = list(np.random.permutation(len(self.train_paths)))
            mini_batches=[]
            for k in range(num_complete_minibatches):
                mini_batches.append(self.data_loader.train_data_loader(k))
            epoch_cost = 0
            epoch_acc = 0
            
            for minibatch in mini_batches:
                (minibatch_x, minibatch_y) = minibatch
                _, minibatch_cost, p, minibatch_acc  = self.sess.run([optim, loss, pred, accuracy], feed_dict = {x: minibatch_x, y: minibatch_y})
                # print("pred shape: ", p)
                epoch_cost += minibatch_cost / num_complete_minibatches
                epoch_acc += minibatch_acc / num_complete_minibatches

            # if epoch%display_step == 0:
            #     (x_val, y_val) = val_data
            #     accuracy_m.eval({x: x_val, y: y_val})
            if epoch%display_step ==0:
                print("cost after epoch %i :  %.3f" % (epoch + 1, epoch_cost), end="")
                print("  train accuracy   :  %.3f" % epoch_acc)
                print("  val accuracy   :  %.3f" % (accuracy.eval({x: x_val, y: y_val})))



            elif epoch%validation_step ==0:
                accuracy.eval({x: x_val, y: y_val})
            



            elif epoch%checkpoint_step ==0:

            

                saver = tf.train.Saver(tf.global_variables(),max_to_keep=self.max_to_keep)
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir))== False:
                    saver.save(self.sess, os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints')))
                else:
                    saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints'))

                # saver.save(sess, '/base_dir/chekpoints.ckpt')

                # with tf.variable_scope("p", reuse=True):
                #         weights = tf.get_variable("kernel")




            elif epoch%summary_step ==0:
                # writer = tf.summary.create_file_writer("/tmp/mylogs")
                # with writer.as_default():
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir))== False:
                    tf.summary.FileWriter(os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries')), self.sess.graph)
                else:
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                # writer = tf.summary.FileWriter('./base_dir/summary', sess.graph)
                # for step in range(100):
                #     # other model code would go here
                #     tf.summary.scalar("my_metric", 0.5, step=step)
                #     writer.flush()
            

          
        print("network trained")
        # predicts = tf.argmax(pred, 1).eval({x:x_test})
        # probs = tf.nn.softmax(pred, 1).eval({x: x_test})
        # # print("test shape: ", x_test.shape)
        # print("predicts shape: ", predicts.shape)
        # print("predicts val: ", predicts)
        # return predicts, probs
#preds , probs = train(x_train, y_train, x_cv, y_cv, x_test, num_epochs=30)

    
    #tf.summary
     #   



    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')


