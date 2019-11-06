import os
import glob
import numpy as np
from skimage.io import imread
# from PIL import imageio
# from PIL import Image
# import image
#.io import imread
import random
import tensorflow as tf
# from keras.utils.np_utils import to_categorical


class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
            test_batch_size, height_of_image, width_of_image, num_channels, num_classes):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes=num_classes

    def load_image(self, path):
        # image=imread(path)
        image=imread(path)
        image=image.reshape(784)
        # image=resize(image, (height_of_image,width_of_image ))
        target_vector=int(os.path.basename(os.path.dirname(path)))

        label=np.eye(self.num_classes)[int(target_vector)]
        
        # label=np.identity(self.num_classes)[int(target_vector)]

        #permutation = list(np.random.permutation(image.shape[0]))
        #shuffled_image = image[permutation]
        #shuffled_label = label[permutation]
        return image , label
        

    def batch_data_loader(self, batch_size, file_paths, index):
        # if len(file_paths)%batch_size !=0:
        #     k=int(len(file_paths)-(len(file_paths)%batch_size))
        #     file_paths=file_paths[:k+1]
        images=[]
        labels=[]
        for i in range(int(index*batch_size),int((index+1)*batch_size)):
            image,label=self.load_image(file_paths[i])
            images.append(image)
            labels.append(label)
        # def one_hot_matrix(labels, C=10):
        #     C = tf.constant(C, name="C")
        #     one_hot_matrix = tf.one_hot(labels, C, axis=1)
        #     with tf.Session() as sess:
        #         one_hot = sess.run(one_hot_matrix)
        # labels=one_hot_matrix(labels, C=10)

        # labels_matrix = np.array(labels)
        # labels_matrix_one_hot = to_categorical(labels_matrix, num_classes=num_classes)

        return images , labels

    def on_epoch_end(self):
        np.random.shuffle(self.train_paths)

    def train_data_loader(self, index):
        # permutation = list(np.random.permutation(len(self.train_paths)))
        # shuffled_paths = self.train_paths[permutation]
        # if index==0:
        #     self.train_paths= shuffled_paths   
        # if index== 0:
        #     random.shuffle(self.train_paths)
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)



