import pandas as pd
import numpy as np
import configparser

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

	train = pd.read_csv(TRAIN_TEST_PATH_CSV + TRAIN_CSV_NAME)
	test = pd.read_csv(TRAIN_TEST_PATH_CSV + TEST_CSV_NAME)

	print('Train normalization started')
	y = train.iloc[:, :1]
	X = train.drop(columns=['label'])
	X = X.astype(np.float)
	X = X / 255.0
	pd.concat([y, X], axis=1).to_csv(TRAIN_TEST_PATH_CSV + TRAIN_CSV_NAME, index=False)
	print('Train normalization finished')

	print('Test normalization started')
	y = test.iloc[:, :1]
	X = test.drop(columns=['label'])
	X = X.astype(np.float)
	X = X / 255.0
	pd.concat([y, X], axis=1).to_csv(TRAIN_TEST_PATH_CSV + TEST_CSV_NAME, index=False)
	print('Test normalization finished')
	
if __name__ == "__main__":
    main()