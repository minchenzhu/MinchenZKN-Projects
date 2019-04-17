import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
# elu
from tensorflow.image import resize_bicubic

# useful macros
# variables
def weight_variable( name, shape, stddev=0.01 ):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial, name=name )

def bias_variable( name, shape ):
  initial = tf.constant(0.01, shape=shape )
  return tf.Variable(initial, name=name )

# def resnet_relu( input ):
#   return lrelu( input )

def resnet_elu( input ):
    return lrelu(input)
  # return elu( input )

def batch_norm( input, phase, params, scope ):
  return tf.contrib.layers.batch_norm( input,
                                       center=params['batch_norm_center'],
                                       scale=params['batch_norm_scale'],
                                       decay=params['batch_norm_decay'],
                                       zero_debias_moving_mean=params['batch_norm_zero_debias'],
                                       is_training=phase,
                                       scope = scope, reuse=tf.AUTO_REUSE )

def bn_dense_discriminator( input, size_in, size_out, name ):
  stddev = np.sqrt( 2.0 / np.float32( size_in + size_out ))
  W = weight_variable( name + '_W', [ size_in, size_out ], stddev )
  b = bias_variable( name + '_bias', [ size_out ] )
  return tf.matmul( input, W ) + b


# moved to simpler resnet layers, only one conv, no relu after identity
class layer_conv3d:
    def __init__(self, layer_id, variables, input, phase, params, no_relu=False):
        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()
            self.output_shape = variables.encoder_W.shape.as_list()

            # if output and input depth differ, we connect them via a 1x1 kernel embedding/projection
            # transformation.
            if variables.resample:
                identity = tf.nn.conv3d(input, variables.encoder_embedding, strides=variables.stride, padding='SAME')
            else:
                identity = input

            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            # self.bn = tf.pad(self.bn, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            self.conv = tf.nn.conv3d(self.bn, variables.encoder_W, strides=variables.stride, padding='SAME')
            if no_relu:
                self.features = self.conv + variables.encoder_b
                print('meow: no relu')
            else:
                self.features = resnet_elu(self.conv + variables.encoder_b)

            self.out = self.features + identity
            self.output_shape = self.out.shape.as_list()



class layer_conv2d:
    def __init__(self, layer_id, variables, input, phase, params, no_relu=False):
        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()
            self.output_shape = variables.encoder_W.shape.as_list()

            # if output and input depth differ, we connect them via a 1x1 kernel embedding/projection
            # transformation.
            if variables.resample:
                identity = tf.nn.conv2d(input, variables.encoder_embedding, strides=variables.stride, padding='SAME')
            else:
                identity = input

            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            # self.bn = tf.pad(self.bn, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            self.conv = tf.nn.conv2d(self.bn, variables.encoder_W, strides=variables.stride, padding='SAME')
            if no_relu:
                self.features = self.conv + variables.encoder_b
                print('meow: no relu')
            else:
                self.features = resnet_elu(self.conv + variables.encoder_b)

            self.out = self.features + identity
            self.output_shape = self.out.shape.as_list()




class layer_pure_conv3D:
    def __init__(self, layer_id, layout, input, phase, params, no_relu = False):
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = input.shape.as_list()
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]

            self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
            self.stddev = np.sqrt(2.0 / self.n)

            self.encoder_W = weight_variable('encoder_W', [self.shape[0],self.shape[1],self.shape[2],self.C_in, self.C_out], self.stddev)
            self.encoder_b = bias_variable('encoder_b', [self.C_out])
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv3d(self.bn, self.encoder_W, strides=self.stride, padding='VALID')
            if no_relu:
                self.features = self.conv + self.encoder_b
                print('meow: no relu')
            else:
                self.features = resnet_elu(self.conv + self.encoder_b)
                # self.features = tf.nn.relu(self.conv + self.encoder_b)


            self.out = self.features
            self.output_shape = self.out.shape.as_list()


class layer_concat:
  def __init__(self, input, batch_size, nviews):
      self.input_shape = input.shape.as_list()
      self.output_shape = [batch_size, nviews, self.input_shape[2], self.input_shape[3], self.input_shape[4]]
      self.shape = [3,3,3,self.input_shape[4],self.input_shape[4]]
      self.n = self.shape[0] * self.shape[1] * self.shape[2]*self.shape[3] + self.shape[4]
      self.stddev = np.sqrt(2.0 / self.n)
      self.concat_W = weight_variable('concat_W', self.shape,self.stddev )
      if nviews == 5:
        strides = [1, 2, 1, 1, 1]
      elif nviews == 9:
        strides = [1, 3, 1, 1, 1]
      self.out = tf.nn.conv3d_transpose(input,self.concat_W,output_shape=self.output_shape,
                                                  strides=strides,
                                                  padding='SAME')


class layer_pure_conv2D:
    def __init__(self, layer_id, layout, input, phase, params, no_relu = False):
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = input.shape.as_list()
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]

            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)
            self.encoder_W = weight_variable('encoder_W', self.shape, self.stddev)
            self.encoder_b = bias_variable('encoder_b', [self.shape[3]])
            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv2d(self.bn, self.encoder_W, strides=self.stride, padding='SAME')
            if no_relu:
                self.features = self.conv + self.encoder_b
                print('meow: no relu')
            else:
                self.features = resnet_elu(self.conv + self.encoder_b)
                # self.features = tf.nn.relu(self.conv + self.encoder_b)


            self.out = self.features
            self.output_shape = self.out.shape.as_list()


def _upsample_along_axis(volume, axis, stride, mode='COPY'):

  shape = volume.get_shape().as_list()

  assert mode in ['COPY', 'ZEROS']
  assert 0 <= axis < len(shape)

  target_shape = shape[:]
  target_shape[axis] *= stride
  target_shape[0] = -1

  padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
  parts = [volume] + [padding for _ in range(stride - 1)]
  volume = tf.concat(parts, min(axis+1, len(shape)-1))

  volume = tf.reshape(volume, target_shape)
  return volume


class layer_decoder_upconv3d:
    def __init__(self, layer_id, variables, batch_size, output_shape, input, phase, params, shared_variables=True):

        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()
            self.output_shape = [batch_size, self.input_shape[1]+2, output_shape[0]+2, output_shape[1]+2, variables.C_out]

            self.bn = batch_norm(input, phase, params, 'batchnorm_input')

            if shared_variables:
                # main decoder pipe, variables are shared between horizontal and vertical
                if variables.stride[2] != 1 or variables.stride[3] != 1:
                    self.bn = tf.stack([resize_bicubic(self.bn[:,i,:,:,:], output_shape) for i in
                           range(0,self.input_shape[1])], axis = 1)


                self.conv = tf.nn.conv3d_transpose(self.bn, variables.decoder_variables_W,
                                                   output_shape=self.output_shape,
                                                   strides=[1, 1, 1, 1, 1], padding='VALID')
                if variables.stride[1] > 1:
                    self.conv = self.conv[:,:,1:-1,1:-1,:]
                else:
                    self.conv = self.conv[:, 1:-1, 1:-1, 1:-1, :]

                self.features = resnet_elu(self.conv + variables.decoder_variables_b)

            if variables.resample:
                # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
                # instead, we use upsampling + conv3d_transpose without stride.
                self.input_upsampled = input
                if variables.stride[2] != 1 or variables.stride[3] != 1:
                    self.input_upsampled = tf.stack(
                          [resize_bicubic(input[:,i,:,:,:], output_shape) for i in
                           range(0,self.input_shape[1])], axis = 1)

                self.input_upsampled = tf.pad(self.input_upsampled, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
                self.input_upsampled = tf.pad(self.input_upsampled, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                identity = tf.nn.conv3d_transpose(self.input_upsampled,
                                                  variables.decoder_variables_embedding,
                                                  output_shape=self.output_shape,
                                                  strides=[1, 1, 1, 1, 1],
                                                  padding='VALID')
                if variables.stride[1] > 1:
                    identity = identity[:,:,1:-1,1:-1,:]
                else:
                    identity = identity[:, 1:-1, 1:-1, 1:-1, :]

            else:
                identity = input

            self.out = self.features + identity
            self.out = self.features


class layer_upconv2d_v2:
    def __init__(self, layer_id, variables, batch_size, input, phase, params, out_channels=-1, no_relu=False):

        with tf.variable_scope(layer_id):

            # define in/out shapes

            self.shape = variables.shape

            self.stride = variables.stride

            # self.input_shape = input.shape.as_list()
            self.input_shape = input.shape

            self.C_in = variables.C_in

            self.C_out = variables.C_out

            self.output_shape = input.shape.as_list()

            self.output_shape[0] = batch_size

            self.output_shape[1] = self.output_shape[1] * self.stride[1]

            self.output_shape[2] = self.output_shape[2] * self.stride[2]

            self.output_shape[3] = self.C_out

            self.W = variables.decoder_variables_W

            self.b = variables.decoder_variables_b

            sh = input.shape.as_list()

            self.resample = variables.resample

            if self.resample:
                self.embedding = variables.decoder_variables_embedding

            if out_channels != -1:
                # output channel override

                self.output_shape[3] = out_channels

                self.shape[2] = out_channels

                self.C_in = out_channels

                self.resample = True

            # generate layers

            self.bn = batch_norm(input, phase, params, 'batchnorm_input')

            if self.stride[1] != 1 or self.stride[2] != 1:
                self.bn = resize_bicubic(self.bn, [np.int(sh[1] * 2), np.int(sh[2] * 2)])

            self.conv = tf.nn.conv2d(self.bn, self.W, strides=[1, 1, 1, 1], padding='SAME')

            if no_relu:

                self.features = self.conv + self.b

            else:

                self.features = resnet_elu(self.conv + self.b)

            if variables.resample:

                if self.stride[1] != 1 or self.stride[2] != 1:

                    self.input_upsampled = resize_bicubic(input, [np.int(sh[1] * 2), np.int(sh[2] * 2)])

                else:

                    self.input_upsampled = input

                identity = tf.nn.conv2d(self.input_upsampled,

                                        self.embedding,

                                        strides=[1, 1, 1, 1],

                                        padding='SAME')




            else:

                identity = input

            self.out = self.features + identity



class layer_decoder_upconv2d:
    def __init__(self, layer_id, variables, batch_size, output_shape, input, phase, params, shared_variables=True):

        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()
            self.output_shape = [batch_size, output_shape[0]+2, output_shape[1]+2, variables.C_out]

            self.bn = batch_norm(input, phase, params, 'batchnorm_input')

            if shared_variables:
                # main decoder pipe, variables are shared between horizontal and vertical
                if variables.stride[1] != 1 or variables.stride[2] != 1:
                    self.bn = resize_bicubic(self.bn, output_shape)

                self.conv = tf.nn.conv2d_transpose(self.bn, variables.decoder_variables_W,
                                                   output_shape=self.output_shape,
                                                   strides=[1, 1, 1, 1], padding='VALID')

                self.conv = self.conv[:,1:-1,1:-1,:]


                self.features = resnet_elu(self.conv + variables.decoder_variables_b)

            if variables.resample:
                # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
                # instead, we use upsampling + conv3d_transpose without stride.
                self.input_upsampled = input
                if variables.stride[1] != 1 or variables.stride[2] != 1:
                    self.input_upsampled = resize_bicubic(input, output_shape)

                self.input_upsampled = tf.pad(self.input_upsampled, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                identity = tf.nn.conv2d_transpose(self.input_upsampled,
                                                  variables.decoder_variables_embedding,
                                                  output_shape=self.output_shape,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID')
                # if variables.stride[1] > 1:
                identity = identity[:,1:-1,1:-1,:]

            else:
                identity = input

            self.out = self.features + identity
            self.out = self.features

class layer_upscale_upconv3d:
    def __init__(self, layer_id, variables, batch_size, output_shape, input, phase, params, shared_variables=True):

        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()

            self.bn = batch_norm(input, phase, params, 'batchnorm_input')

            if shared_variables:
                # main decoder pipe, variables are shared between horizontal and vertical
                if variables.stride[2] != 1 or variables.stride[3] != 1:
                    self.bn = tf.stack([resize_bicubic(self.bn[:,i,:,:,:], output_shape) for i in
                           range(0,self.input_shape[1])], axis = 1)

                self.bn = tf.pad(self.bn, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
                self.bn = tf.pad(self.bn, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                self.conv = tf.nn.conv3d(self.bn, variables.decoder_variables_W,
                                                   strides=[1,1,1,1,1], padding='VALID')
                self.features = resnet_elu(self.conv + variables.decoder_variables_b)

            if variables.resample:
                # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
                # instead, we use upsampling + conv3d_transpose without stride.
                self.input_upsampled = input
                if variables.stride[2] != 1 or variables.stride[3] != 1:
                    self.input_upsampled = tf.stack(
                          [resize_bicubic(input[:,i,:,:,:], output_shape) for i in
                           range(0,self.input_shape[1])], axis = 1)

                identity = tf.nn.conv3d(self.input_upsampled,
                                                  variables.decoder_variables_embedding,
                                                  strides=[1, 1, 1, 1, 1],
                                                  padding='VALID')

            else:
                identity = input

            self.out = self.features + identity


class encoder_variables:
    def __init__(self, layer_id, layout):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = [1,1,layout['stride'][-3],layout['stride'][-2],layout['stride'][-1]]
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []

            # number of channels in/out -> determines need for identity remapping
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]
            self.resample = self.C_in != self.C_out or self.stride[2] != 1 or self.stride[3] != 1
            if self.resample:
                self.project_n = self.C_in * self.stride[2] * self.stride[3] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.encoder_embedding = weight_variable('encoder_embedding', [1, 1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
            self.stddev = np.sqrt(2.0 / self.n)

            self.encoder_W = weight_variable('encoder_W_3D', self.shape, self.stddev)
            self.encoder_b = bias_variable('encoder_b_3D', [self.shape[4]])


class encoder_variables_2D:
    def __init__(self, layer_id, layout):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = [1,layout['stride'][1],layout['stride'][-2],layout['stride'][-1]]
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []

            # number of channels in/out -> determines need for identity remapping
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]
            self.resample = self.C_in != self.C_out or self.stride[2] != 1 or self.stride[3] != 1
            if self.resample:
                self.project_n = self.C_in * self.stride[2] * self.stride[3] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.encoder_embedding = weight_variable('encoder_embedding_2D', [1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.encoder_W = weight_variable('encoder_W_2D', self.shape, self.stddev)
            self.encoder_b = bias_variable('encoder_b_2D', [self.shape[-1]])


class decoder_variables_3D:
    def __init__(self, layer_id, layout, i, last_layer, patch_weight):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = []
            self.pinhole_weight = []
            if i == last_layer - 1:
                self.C_in = self.shape[-1]
                self.C_out = self.shape[-2]*patch_weight
            else:
                self.C_out = self.shape[-2] * patch_weight

                # if skip_connection:
                #     self.C_in = self.shape[-1] * (patch_weight+1)
                # else:
                self.C_in = self.shape[-1] * patch_weight




            self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1 or self.stride[3] != 1

            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2] * self.stride[3] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.decoder_variables_embedding = weight_variable('decoder_variables_embedding', [1, 1, 1, self.C_out, self.C_in],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
            self.stddev = np.sqrt(2.0 / self.n)

            self.decoder_variables_W = weight_variable('decoder_variables_W',
                                           [self.shape[0], self.shape[1], self.shape[2], self.C_out, self.C_in], self.stddev)
            self.decoder_variables_b = bias_variable('decoder_variables_b', [self.C_out])



class decoder_variables:
    def __init__(self, layer_id, layout, i, last_layer, patch_weight):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = []
            self.pinhole_weight = []
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]


            self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1

            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2] * self.stride[3] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.decoder_variables_embedding = weight_variable('decoder_variables_embedding', [1, 1,  self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.decoder_variables_W = weight_variable('decoder_variables_W',
                                           [self.shape[0], self.shape[1],  self.C_in, self.C_out], self.stddev)
            self.decoder_variables_b = bias_variable('decoder_variables_b', [self.C_out])





class upscale_variables:
    def __init__(self, layer_id, layout, interpolate, decoder_channels, patch_weight):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []

            # number of channels in/out -> determines need for identity remapping
            if interpolate:
                self.C_in = np.int(self.shape[-1])*patch_weight + decoder_channels
            else:
                self.C_in = self.shape[-1]*patch_weight
            self.C_out = self.shape[-2]*patch_weight
            self.resample = self.C_in != self.C_out or self.stride[1] !=1 or self.stride[2] !=1 or self.stride[3] !=1
            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2] * self.stride[3] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.decoder_variables_embedding = weight_variable('upscale_embedding', [1, 1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
            self.stddev = np.sqrt(2.0 / self.n)

            self.decoder_variables_W = weight_variable('upscale_W', [self.shape[0], self.shape[1], self.shape[2], self.C_in, self.C_out], self.stddev)
            self.decoder_variables_b = bias_variable('upscale_b', [self.C_out])

class discriminator_variables_2D:
    def __init__(self, layer_id, layout):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []

            # number of channels in/out -> determines need for identity remapping
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]
            self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2] + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.discriminator_embedding = weight_variable('discriminator_embedding', [1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.discriminator_W = weight_variable('discriminator_W', self.shape, self.stddev)
            self.discriminator_b = bias_variable('discriminator_b', [self.shape[3]])


class discriminator_conv2d:
    def __init__(self, layer_id, variables, input, no_relu=False):
        with tf.variable_scope(layer_id):

            self.input_shape = input.shape.as_list()
            self.output_shape = variables.discriminator_W.shape.as_list()

            # if output and input depth differ, we connect them via a 1x1 kernel embedding/projection
            # transformation.
            if variables.resample:
                identity = tf.nn.conv2d(input, variables.discriminator_embedding, strides=variables.stride, padding='SAME')
            else:
                identity = input

            # input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            self.conv = tf.nn.conv2d(input, variables.discriminator_W, strides=variables.stride, padding='SAME')
            if no_relu:
                self.features = self.conv + variables.discriminator_b
                print('meow: no relu')
            else:
                self.features = resnet_elu(self.conv + variables.discriminator_b)

            self.out = self.features + identity
            self.output_shape = self.out.shape.as_list()



def bn_dense_discriminator_variables(size_in, size_out, name ):
  stddev = np.sqrt( 2.0 / np.float32(size_in + size_out))
  W = weight_variable( name + '_W', [ size_in, size_out ], stddev )
  b = bias_variable( name + '_bias', [ size_out ] )
  return (W ,b)

def bn_dense_discriminator( input, W, b):
  # stddev = np.sqrt( 2.0 / np.float32( size_in + size_out ))
  # W = weight_variable( name + '_W', [ size_in, size_out ], stddev )
  # b = bias_variable( name + '_bias', [ size_out ] )
  return tf.matmul( input, W ) + b


def pinhole_conv3d(variables, input):

  conv = tf.nn.conv3d( input, variables.pinhole_weight, strides=[1,1,1,1,1], padding='SAME' )
  return conv

def pinhole_weight(variables, input):
  shape = input.shape.as_list()

  pinhole_weight = weight_variable('pinhole_weight', [1, 1, 1, shape[-1], variables.C_out],
                                   stddev = variables.stddev)
  return pinhole_weight