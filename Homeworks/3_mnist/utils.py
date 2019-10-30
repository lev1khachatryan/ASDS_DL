import os
import numpy as np
def create_dir(dir_name, relative_path):
    """
    Create new directory if not exists
    
    Parameters:
        dir_name (string)      - name of directory we want to create
        relative_path (string) - absolute path of directory we want to create
        
    Returns:
        path (string)          - full path of directory
    """
    
    path = relative_path + dir_name
    if not os.path.exists(path):
        os.mkdir(path)
    return path

# function to normalize data
def normalize_data(data):
    # scale features using statistics that are robust to outliers
    #rs = sklearn.preprocessing.RobustScaler()
    #rs.fit(data)
    #data = rs.transform(data)
    #data = (data-data.mean())/(data.std()) # standardisation
    data = data / data.max() # convert from [0:255] to [0.:1.]
    #data = ((data / 255.)-0.5)*2. # convert from [0:255] to [-1.:+1.]
    return data

# convert one-hot encodings into labels
def one_hot_to_dense(labels_one_hot):
    return np.argmax(labels_one_hot,1)