import os
import glob
from skimage.io import imread
import numpy as np

class DataLoader:

    def __init__(self, train_images_dir, val_images_dir, test_images_dir, train_batch_size, val_batch_size, 
            test_batch_size, height_of_image, width_of_image, num_channels, num_classes):

        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes

        
    def load_image(self, path):
        matrix = imread(path)
        matrix = matrix.reshape(784)
        label = np.identity(10)[int(path.split('/')[-2:][0])]
        return(matrix, label)
        pass

    def batch_data_loader(self, batch_size, file_paths, index):
        images = []
        labels = []   
        for image in file_paths[index*batch_size : (index+1)*batch_size]:
          images.append(self.load_image(image)[0])
          labels.append(self.load_image(image)[1])
        return(images, labels)
        pass

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)
      
#     def fcount(self, path, exts=[".jpg"]):
#         count=0    
#         exts=[e.lower() for e in exts]
#         for root, dirs, files in os.walk(path):
#             for d in dirs:
#                 p=os.path.join(root, d)
#                 ff=[fn for fn in os.listdir(p) if any(fn.lower().endswith(e) for e in exts) ]
#                 if ff:
#                     count+=len(ff)
#         return count
#         pass  
      
    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.testpaths, index)


