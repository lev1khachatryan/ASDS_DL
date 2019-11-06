import os
import glob
from skimage.io import imread
#import imageio
#import image
import numpy as np
#from keras.utils.np_utils import to_categorical
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
    	matrix = imread(path)
    	matrix = matrix.reshape(784)
    	label = int(os.path.basename(os.path.dirname(path)))
    	label_one = np.eye(self.num_classes)[int(label)]
    	return(matrix, label_one)

    def batch_data_loader(self, batch_size, file_paths, index):
        images_matrix = []
        labels_matrix = []
        for i in file_paths[index*batch_size : (index+1)*batch_size]:
            images = self.load_image(i)[0]
            images_matrix.append(images)
            labels = self.load_image(i)[1]
            labels_matrix.append(labels)

        #labels_matrix = np.array(labels_matrix)
        #labels_matrix_one_hot = to_categorical(labels_matrix, num_classes=num_classes)
	    
        return(images_matrix, labels_matrix)
       

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size,self.train_paths, index)
  
    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)


