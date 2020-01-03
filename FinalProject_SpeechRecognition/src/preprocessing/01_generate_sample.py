import sys
sys.path.insert(1, '../')

from utils import *
from functions import *

import configparser

def main():
	config = configparser.ConfigParser()
	config.read('../config.INI')

	data_dir = config['paths']['DATA_DIR']
	sample_dir  = config['paths']['SAMPLE_DIR']

	dataset = Dataset(data_dir)

	dataset.gen_sample_set(sample_dir=sample_dir)

if __name__ == "__main__":
    main()