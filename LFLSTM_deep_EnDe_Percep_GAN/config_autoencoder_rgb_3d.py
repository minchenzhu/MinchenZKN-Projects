# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'LFLSTM_try_48_2D_EnDe_Deep_Percep_GAN'

data_path = '/home/mz/HD_data/LFLSTM_data/'
tf_log_path = '/home/mz/HD_data/Tensorboard_Logs/'

# data_path = '/data_fast/LFLSTM/'
# tf_log_path = '/data_fast/aa/tf_logs/LSTM/'

# data_path = '/data/aa/LFLSTM_data/'
# tf_log_path = '/data/aa/LSTM_tf_logs/'

# CURRENT TRAINING DATASET
training_data = [
    # data_path + 'lf_patch_benchmark_rgb_sr_s4.hdf5',
    data_path + 'lflstm_patch_try_48_v2.hdf5',
]
# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    'ColorSpace'                  : 'RGB',
    'VisibleGPU'                  :'0,1,2,3',
    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,
    # add interpolated input as patch for skip connection in the upscaling phase
    'interpolate'          : True,
    # this will log every tensor being allocated,
    # very spammy, but useful for debugging
    'log_device_placement' : False,
}

# encoder for 48 x 48 patch, 9 views, RGB
view_proc = '2D'


D = 9
D_in = 2

H = 48 # patch spatial size
W = 48
sx = 16 # patch stepping
sy = 16

C = 3

# Number of features in the layers
L_half = 8
L = 16
L0 = 24
L1 = 32
L2 = 64
L3 = 96
L4 = 128
L5 = 192
L6 = 256
L7 = 384



# fraction between encoded patch and decoded patch. e.g.
# the feature maps of decoded patch are 3 times as many
# as the encoded patch, then patch_weight = 3

# Encoder stack for downwards convolution
patch_weight = 1

h_dim = L4


# chain of dense layers to form small bottleneck (can be empty)
layers = dict()

input_GPU = 0
LSTM_GPU = 1
GAN_GPU = 2



layers['encoder_3D'] = [
    {'conv': [3, 3, 3, C, L0],
     'stride': [1, 1, 1, 1, 1]
     },
    {'conv': [3, 3, 3, L0, L1],
     'stride': [1, 1, 1, 1, 1]
     },
    # resolution now 9 x 18 x 18
    {'conv': [3, 3, 3, L1, L2],
     'stride': [1, 1, 1, 1, 1]
     },
    # resolution now 5 x 18 x 18
    {'conv': [3, 3, 3, L2, L3],
     'stride': [1, 1, 1, 1, 1]
     },
]

layers['encoder_2D'] = [
    {'conv': [3, 3, C * 3, L1],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L1, L2],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L2, L4],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L4, L6],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L6, L7],
     'stride': [1, 1, 1, 1]
     },

]

layers['skip_2D'] = [
    {'conv': [3, 3, L6, L2],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L4, L1],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L2, L0],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L1, L],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, C * 3, L_half],
     'stride': [1, 1, 1, 1]
     },

]

layers['LSTM'] = [
    {'conv': [3, 3, L4 * 3 + h_dim * 2, L4 * 3 + h_dim],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L4 * 3 + h_dim, L4 * 3],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L4 * 3, L4 * 2],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L4 * 2, h_dim],
     'stride': [1, 1, 1, 1]
     },
]

layers['decoder_2D'] = [
    {'conv': [3, 3, h_dim + L2, L4],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L4 + L1, L4],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L4 + L0, L3],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L3 + L, L3],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L3 + L_half, L2],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L2, L1],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L1, L],
     'stride': [1, 1, 1, 1]
     },
    {'conv': [3, 3, L, L_half],
     'stride': [1, 1, 1, 1]
     },
    # # resolution now 5 x 5 x 5
    {'conv': [1, 1, L_half, C],
     'stride': [1, 1, 1, 1]
     },
]

layers['GAN_2D'] = [
    {'conv': [3, 3, C, L],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L, L1],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L1, L2],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L2, L3],
     'stride': [1, 2, 2, 1]
     },
    {'conv': [3, 3, L3, L4],
     'stride': [1, 1, 1, 1]
     },

]


# 3D ENCODERS
encoders_3D = [
    {
        'id': 'RGB',
        'channels': C,
        'preferred_gpu': 0,
    },
]


decoders_3D = [
    {
        'id': 'RGB',
        'preferred_gpu': 2,
        'loss_fn': 'L2',
        'train': True,
        'weight': 1.0,
        'no_relu': False,
        'adv_loss_weight': 1e-3,
    },
]

L2_weight = 1.0
adv_weight = 1e-4


discriminator = [
    {
        'step_size': 1e-4,
        'train': True,
        'weight': 1.0,
        'channels': C,
        'percep_layer': 2,
        'percep_weight' :10,
    }
]

iterGan = 1


# MINIMIZERS
minimizers = [
    # center view super resolution
    {
        'id': 'RGB_min',  # 'YCBCR_min'
        'losses_3D': ['RGB'],  # 'KL_divergence' , 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
        'optimizer': 'Adam',
        'preferred_gpu': 3,
        'step_size': 1e-4,
    },
 ]


# TRAINING HYPERPARAMETERS
training = dict()

# subsets to split training data into
# by default, only 'training' will be used for training, but the results
# on mini-batches on 'validation' will also be logged to check model performance.
# note, split will be performed based upon a random shuffle with filename hash
# as seed, thus, should be always the same for the same file.
#
training[ 'subsets' ] = {
  'validation'   : 0.05,
  'training'     : 0.95,
}


# number of samples per mini-batch
# reduce if memory is on the low side,
# but if it's too small, training becomes ridicuously slow
training[ 'samples_per_batch' ] = 10

# log interval (every # mini-batches per dataset)
training[ 'log_interval' ] = 5

# save interval (every # iterations over all datasets)
training[ 'save_interval' ] = 100

# noise to be added on each input patch
# (NOT on the decoding result)
training[ 'noise_sigma' ] = 0.0

# decay parameter for batch normalization
# should be larger for larger datasets
training[ 'batch_norm_decay' ]  = 0.9
# flag whether BN center param should be used
training[ 'batch_norm_center' ] = False
# flag whether BN scale param should be used
training[ 'batch_norm_scale' ]  = False
# flag whether BN should be zero debiased (extra param)
training[ 'batch_norm_zero_debias' ]  = False

eval_res = {
    'h_mask_s4': 150,
    'w_mask_s4': 150,
    'h_mask_s2': 70,
    'w_mask_s2': 70,
    'm_s4': 32,
    'm_s2': 16,
    'min_mask': 0.1,
    'result_folder': "./results/",
    'test_data_folder': "H:\\testData\\"
}
