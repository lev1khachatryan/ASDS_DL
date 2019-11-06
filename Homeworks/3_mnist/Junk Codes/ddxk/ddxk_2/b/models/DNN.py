from .BaseNN import *
class DNN(BaseNN):
   def network(self, X):
       layer = tf.layers.dense(X, units = self.num_classes, activation = tf.nn.softmax)
       return layer
   def metrics(self, Y, Y_pred):
       correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
       loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_pred, labels = Y))
       accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
       #accuracy = tf.reduce_mean(tf.cast(Y_pred, tf.float32))
       return loss, accuracy

