from .BaseNN import *

class DNN(BaseNN):

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob):

        super().__init__(train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, keep_prob)

        self.n_neurons = 128

    
    def load_tensors(self, graph):
        """
        load tensors from a saved graph
        -----------------
        Parameters:
            graph       (tf.graph_like) - graph we obtained from saved file
        Returns:
            None
        -----------------
        """

        # input tensors
        self.x_data_tf = graph.get_tensor_by_name("x_data_tf:0")
        self.y_data_tf = graph.get_tensor_by_name("y_data_tf:0")
        
        # weights and bias tensors
        self.W_conv1_tf = graph.get_tensor_by_name("W_conv1_tf:0")
        self.W_conv2_tf = graph.get_tensor_by_name("W_conv2_tf:0")
        self.W_conv3_tf = graph.get_tensor_by_name("W_conv3_tf:0")
        self.W_fc1_tf = graph.get_tensor_by_name("W_fc1_tf:0")
        self.W_fc2_tf = graph.get_tensor_by_name("W_fc2_tf:0")
        self.b_conv1_tf = graph.get_tensor_by_name("b_conv1_tf:0")
        self.b_conv2_tf = graph.get_tensor_by_name("b_conv2_tf:0")
        self.b_conv3_tf = graph.get_tensor_by_name("b_conv3_tf:0")
        self.b_fc1_tf = graph.get_tensor_by_name("b_fc1_tf:0")
        self.b_fc2_tf = graph.get_tensor_by_name("b_fc2_tf:0")
        
        # activation tensors
        self.h_conv1_tf = graph.get_tensor_by_name('h_conv1_tf:0')  
        self.h_pool1_tf = graph.get_tensor_by_name('h_pool1_tf:0')
        self.h_conv2_tf = graph.get_tensor_by_name('h_conv2_tf:0')
        self.h_pool2_tf = graph.get_tensor_by_name('h_pool2_tf:0')
        self.h_conv3_tf = graph.get_tensor_by_name('h_conv3_tf:0')
        self.h_pool3_tf = graph.get_tensor_by_name('h_pool3_tf:0')
        self.h_fc1_tf = graph.get_tensor_by_name('h_fc1_tf:0')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        
        # training and prediction tensors
        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
        self.cross_entropy_tf = graph.get_tensor_by_name('cross_entropy_tf:0')
        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        self.y_pred_proba_tf = graph.get_tensor_by_name("y_pred_proba_tf:0")
        self.y_pred_correct_tf = graph.get_tensor_by_name('y_pred_correct_tf:0')
        self.accuracy_tf = graph.get_tensor_by_name('accuracy_tf:0')
        
        # tensor of stored losses and accuracies during training
        self.train_loss_tf = graph.get_tensor_by_name("train_loss_tf:0")
        self.train_acc_tf = graph.get_tensor_by_name("train_acc_tf:0")
        self.valid_loss_tf = graph.get_tensor_by_name("valid_loss_tf:0")
        self.valid_acc_tf = graph.get_tensor_by_name("valid_acc_tf:0")

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

    def network(self, X):
        """
        Construct network architecture
        -----------------
        Parameters:
            X       (tensor) - input data with shape of (batch size; height of image; width of image ; num channels)
        Returns:
            Last layer of the network (for continuation)
        -----------------
        """

        self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.n_neurons)
        self.output, self.state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        self.z_pred_tf = tf.layers.dense(self.state, self.num_classes)

        return self.z_pred_tf

    def metrics(self, Y, Y_pred):
        """
        Some metric, here I use simple accuracy
        -----------------
        Parameters:
            Y       (array_like) - actual labels
            Y_pred  (array_like) - predicted labels
        Returns:
            Float (Accuracy)
        -----------------
        """

        Y = Y.reshape(-1,)
        Y_pred = Y_pred.reshape(-1,)
        return np.mean(Y == Y_pred)