import file_io
import matplotlib.pyplot as plt
import h5py
from scipy.misc import imresize
from skimage import measure
import os
import lf_tools
import numpy as np
import scipy.io as scio
from scipy.misc import imsave

data_dir = "/home/mz/HD_data/LFLSTM_data/lflstm_patch_try_2.hdf5"
f = h5py.File(data_dir, 'r')
lf=f['lf_patches']
for k in range(0,9):
    plt.figure(k)
    plt.imshow(lf[k,4,:,:,:,500])
    plt.imshow(lf[4, k, :, :, :, 1200])