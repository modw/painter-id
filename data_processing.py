import pandas as pd
import numpy as np
# for sampling and processing
from sklearn.feature_extraction.image import extract_patches
from sklearn.preprocessing import RobustScaler


# # Function definitions

# ## Importing and Adjusting

def data_to_array(data_address):
    """Imports x,y,z txt data, grabs and reshapes z values,
    returns ndarray with appropriate shape.
    
    Parameters
    ----------
    data_address: str
        file of xyz data
    
    Returns
    ----------
    data: 2d ndarray with z values"""
    # import data
    data = pd.read_csv(data_address, header=None,
                  names = ['x','y','z'])
    # find out length and width
    width, length = data['x'].nunique(), data['y'].nunique()
    # get z-data as reshaped ndarray 
    z_arr = data['z'].values.reshape(length, width)
    # set negative values no NaNs
    z_arr[z_arr<0] = None
    return z_arr


def scale_data(data, scaler):
    """Takes data and sklearn scaler as input, returns scaled data.
    The function's purpose is to deal with reshaping."""
    # fit scaler to flattened data
    data_scaled = scaler.fit_transform(data.reshape(-1,1))
    # returns scaled data in original dimensions
    return data_scaled.reshape(data.shape)


# ## Sampling


def get_patches(data, patch_size, shift=True, dropna=True):
    """Splits the data uniformly in square patches of 
    size patch_size. Returns new array with each patch.
    
    Parameters
    ----------
    data : ndarray
       2d array of which patches are extracted
    patch_size : integer
        The size of the patch square side
    shift: bool
        If True, subtract off mean of each patch
    dropna: bool
        If True, discard patches with any NaN element
        
    Returns
    ----------
    patches: 3d array of dimensions (n_patches, patch_size, patch_size)
    """
    patches = extract_patches(data, patch_size, 
                              extraction_step=patch_size)
    patches = patches.reshape(-1,patch_size,patch_size)
    if dropna:
        # drop patches with nans in them
        patches = patches[~np.isnan(patches).any(axis=(1,2))]
    if shift:
        # take mean of each patch and reshape it for subtaction
        means = np.mean(patches, axis=(1,2)).reshape(-1,1,1)
        patches -= means
    return patches


def get_X_y(data_list, shuffle=True):
    """
    Parameters
    ---------
    data_tuple: list of arrays
        A list with arrays of patches
    shuffle: bool
        Whether or not to shuffle the elements in X,y
    Returns
    ---------
    X, y: X is an array with the patches and y has the corresponding labels as ints
    """
    y_list = [None]*len(data_list)
    
    for i, data in enumerate(data_list):
        y_list[i] = i*np.ones(len(data))

    y = np.concatenate(y_list).reshape(-1,1)
    X = np.concatenate(data_list)
    if shuffle:
        shuffled_idx = np.arange(len(y))
        np.random.shuffle(shuffled_idx)
        X = X[shuffled_idx]
        y = y[shuffled_idx]
    return X, y


# ## Pipeline

# create a function that combines all the previous steps
# function should take list of file addresses and return X, y
# step 1: convert data to array
# step 2: scale data array, the scaler should be an option
# step 3: get_patches from scaled data, size, shifting and dropping nans should be options
# step 4: produce X and y arrays and return them

def processing_pipeline(files, patch_size, scaler=None,
                       shift=True, dropna=True, shuffle=True):
    """
    Function to process data for training.
    Returns X and y arrays. The returned arrays are of same size
    as the z dimension of the files, but the algorithm needs 3 times
    that amount of memory.
    
    Parameters
    ---------
    files: list of str
        Files of xyz data
    patch_size : integer
        The size of the patch square side
    shift: bool
        If True, subtract off mean of each patch
    dropna: bool
        If True, discard patches with any NaN element
    scaler: scikit-learn scaler Class object; defaults to None
        Scaler to scale the data. Defaults to RobustScaler
    shuffle: bool
        Whether or not to shuffle the elements in X,y
   
    Returns
    ---------
    X, y: X is an array with the patches and y has the corresponding labels as ints
    """
    # create lists to store data
    data_arrs = [None]*len(files)
    patches_list = [None]*len(files)
    # loop over files
    for i in range(len(files)):
        # save data to 2d array
        data_arrs[i] = data_to_array(files[i])
        # scale arrays inplace
        if scaler:
            data_arrs[i] = scale_data(data_arrs[i], scaler)
        # get patches
        patches_list[i] = get_patches(data_arrs[i], patch_size,
                                     shift, dropna)
    # get X,y arrays
    X, y = get_X_y(patches_list, shuffle)
    return X,y