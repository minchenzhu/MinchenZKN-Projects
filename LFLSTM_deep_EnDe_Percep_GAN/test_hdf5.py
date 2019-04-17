import numpy as np
import scipy.io as sc
import h5py
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from file_io import get_dataset_as_type
import lf_tools
import file_io


# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_1.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_2.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_1.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_2.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_3.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_4.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_5.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_disp_half_6.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_flowers_1.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_HCI.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_synthetic_rgb_sr_s4_stanford.hdf5'
# data_path = '/home/mz/HD_data/SR_data_backups/Data_Cross_HR/lf_patch_benchmark_rgb_sr_s4.hdf5'
# data_path = '/home/mz/HD_data/CVPR_Sup_Mat/light_field_512/lf_patch_light_field_half.hdf5'
data_path = '/home/mz/HD_data/LFLSTM_data/lflstm_patch_try_48_v2.hdf5'
# data_path = '/home/mz/HD_data/CVPR_Sup_Mat/new_light_field/lf_patch_synthetic_new_1024.hdf5'





f = h5py.File(data_path, 'r')

lf = f['lf_patches']

k = 0

