import csv
import os
import imageio
import numpy as np
import pandas as pd
import configparser
from utils import *
from PIL import Image
import scipy.misc
from sklearn.model_selection import train_test_split


def main():
	config = configparser.ConfigParser()
	config.read('config.INI')

	TRAIN_PATH_IMAGE = config['paths']['TRAIN_PATH_IMAGE']
	TEST_PATH_IMAGE  = config['paths']['TEST_PATH_IMAGE']
	VALIDATION_PATH_IMAGE  = config['paths']['VALIDATION_PATH_IMAGE']

	TRAIN_TEST_PATH_CSV = config['paths']['TRAIN_TEST_PATH_CSV']

	TRAIN_CSV_NAME = config['file_names']['TRAIN_CSV_NAME']
	TEST_CSV_NAME  = config['file_names']['TEST_CSV_NAME']
	VALIDATION_CSV_NAME  = config['file_names']['VALIDATION_CSV_NAME']

	IMG_WIDTH = int(config['image_shape']['IMG_WIDTH'])
	IMG_HEIGHT = int(config['image_shape']['IMG_HEIGHT'])

	PERCENT_OF_VALSET = float(config['other']['PERCENT_OF_VALSET'])

	print('From Train data to Train - Validation')
	train = pd.read_csv(TRAIN_TEST_PATH_CSV + TRAIN_CSV_NAME)

	y = train.iloc[:, :1]
	X = train.drop(columns=['label'])
	
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=PERCENT_OF_VALSET, random_state=22, shuffle=True)
	
	Train = pd.concat([y_train, X_train], axis=1)
	Validation = pd.concat([y_val, X_val], axis=1)
	
	Validation.to_csv(TRAIN_TEST_PATH_CSV + VALIDATION_CSV_NAME, index=False)
	Train.to_csv(TRAIN_TEST_PATH_CSV + TRAIN_CSV_NAME, index=False)

	# image id as an image name 
	image_id = 1

	print("Train data preprocessing started")

	with open(TRAIN_TEST_PATH_CSV + TRAIN_CSV_NAME) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    
	    # first row contains only column names
	    first_row = 1
	    
	    for row in csv_reader:
	        
	        if first_row:
	            first_row = 0
	            continue
	            
	        label = str(row[0])
	        array = np.array(row[1:], dtype=np.float).reshape((IMG_WIDTH, IMG_HEIGHT))
	        
	        path = create_dir(label, TRAIN_PATH_IMAGE)
	        
	        # Image.fromarray(array).save(path + '\\' + f"{image_id}.png")
	        scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(path + '\\' + f"{image_id}.png")
	        
	        image_id += 1

	print("Train data preprocessing finished")

	print("Test data preprocessing started")

	with open(TRAIN_TEST_PATH_CSV + TEST_CSV_NAME) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    
	    # first row contains only column names
	    first_row = 1
	    
	    for row in csv_reader:
	        
	        if first_row:
	            first_row = 0
	            continue
	            
	        label = str(row[0])
	        array = np.array(row[1:], dtype=np.float).reshape((IMG_WIDTH, IMG_HEIGHT))
	        
	        path = create_dir(label, TEST_PATH_IMAGE)
	        
	        # Image.fromarray(array).save(path + '\\' + f"{image_id}.png")
	        scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(path + '\\' + f"{image_id}.png")
	        
	        image_id += 1

	print("Test data preprocessing finished")
	
	print("Validation data preprocessing started")

	with open(TRAIN_TEST_PATH_CSV + VALIDATION_CSV_NAME) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    
	    # first row contains only column names
	    first_row = 1
	    
	    for row in csv_reader:
	        
	        if first_row:
	            first_row = 0
	            continue
	            
	        label = str(row[0])
	        array = np.array(row[1:], dtype=np.float).reshape((IMG_WIDTH, IMG_HEIGHT))
	        
	        path = create_dir(label, VALIDATION_PATH_IMAGE)
	        
	        # Image.fromarray(array).save(path + '\\' + f"{image_id}.png")
	        scipy.misc.toimage(array, cmin=0.0, cmax=1.0).save(path + '\\' + f"{image_id}.png")
	        
	        image_id += 1

	print("Validation data preprocessing finished")

if __name__ == "__main__":
    main()