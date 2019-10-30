train_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/train/image/'
val_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/validation/image/'
test_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/test/image/'
num_epochs=2
train_batch_size=100
val_batch_size=100
test_batch_size=100
height_of_image=28
width_of_image=28
num_channels=1
num_classes=10
learning_rate = 0.001
base_dir='./results'
max_to_keep=5
model_name='nn_1'

data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)

x_train, y_train = data_loader.all_train_data_loader()
x_valid, y_valid = data_loader.all_val_data_loader()

x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)

# start timer
start = datetime.datetime.now();

nn_graph = DNN(
    train_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/train/image/',
    val_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/validation/image/',
    test_images_dir='C:/_Files/MyProjects/ASDS_3/ASDS_DL/Homeworks/3_mnist/_inputs_image/all/test/image/',
    num_epochs=2,
    train_batch_size=100,
    val_batch_size=100,
    test_batch_size=100,
    height_of_image=28,
    width_of_image=28,
    num_channels=1,
    num_classes=10,
    learning_rate = 0.001,
    base_dir='./results',
    max_to_keep=5,
    model_name='nn_1'
)

nn_graph.create_network() # create graph
nn_graph.attach_saver() # attach saver tensors

# start tensorflow session
with tf.Session() as sess:

    # attach summaries
    nn_graph.attach_summary(sess) 

    # variable initialization of the default graph
    sess.run(tf.global_variables_initializer()) 

    # training on original data
    nn_graph.train_graph_helper(sess, x_train, y_train, x_valid, y_valid, n_epoch = 1.0)

    # training on augmented data
#     nn_graph.train_graph_helper(sess, x_train, y_train, x_valid, y_valid, n_epoch = 14.0,
#                         train_on_augmented_data = True)

    # save tensors and summaries of model
    nn_graph.save_model(sess)

print('total running time for training: ', datetime.datetime.now() - start)
