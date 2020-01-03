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

	use_sample = float(config['other']['USE_SAMPLE'])

	if use_sample == 1:
		audio_path = sample_dir + '/train/audio/'
		pict_Path  = sample_dir + '/train/pics/'
		test_audio_path = sample_dir + '/test/audio/'
		test_pict_Path  = sample_dir + '/test/pics/'
	elif use_sample == 0:
		audio_path = data_dir + '/train/audio/'
		pict_Path  = data_dir + '/train/pics/'
		test_audio_path = data_dir + '/test/audio/'
		test_pict_Path  = data_dir + '/test/pics/'
	samples = []

	# subFolderList = []
	# for x in os.listdir(audio_path):
	#     if os.path.isdir(audio_path + '/' + x):
	#         subFolderList.append(x)
	if not os.path.exists(pict_Path):
	    os.makedirs(pict_Path)
	if not os.path.exists(test_pict_Path):
	    os.makedirs(test_pict_Path)
	subFolderList = []
	for x in os.listdir(audio_path):
	    if os.path.isdir(audio_path + '/' + x):
	        subFolderList.append(x)
	        if not os.path.exists(pict_Path + '/' + x):
	            os.makedirs(pict_Path +'/'+ x)

	sample_audio = []
	total = 0
	for x in subFolderList:
	    
	    # get all the wave files
	    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
	    total += len(all_files)
	    # collect the first file from each dir
	    sample_audio.append(audio_path  + x + '/'+ all_files[0])
	    
	    # show file counts
	    print('count: %d : %s' % (len(all_files), x ))
	print(total)

	for i, x in enumerate(subFolderList):
	    print(i, ':', x)
	    # get all the wave files
	    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
	#     for file in all_files[:10]:
	    for file in all_files:
	        wav2img(audio_path + x + '/' + file, pict_Path + x)

if __name__ == "__main__":
    main()