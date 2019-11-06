from .BaseNN import *


class DNN(BaseNN):
  
    def network(self, X):
        l_softmax = tf.layers.dense(X, units=10,  activation=tf.nn.softmax)
        return l_softmax
      
    def metrics(self, Y, Y_pred):
        correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_pred, labels = Y))
#         optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        return cost, accuracy 