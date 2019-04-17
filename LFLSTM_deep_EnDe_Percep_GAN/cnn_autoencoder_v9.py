# Class definition for the combined CRY network
# drops: deep regression on angular patch stacks
#
# in this version, we take great care to have nice
# variable scope names.
#
# start session
import code
import tensorflow as tf
import numpy as np
import math
import libs.layers as layers
import libs.operations as ops
from tensorflow.image import resize_bilinear
import tensorflow.contrib.slim as slim


# main class defined in module
class create_cnn:

  def __init__( self, config ):

    # config (hyperparameters)
    self.config = config
    self.max_layer = config.config[ 'max_layer' ]
    self.interpolate = config.config['interpolate']

    # both stacks have 9 views, patch size 16x16 + 16 overlap on all sides,
    # for a total of 48x48.
    self.C = config.C
    self.D = config.D
    self.D_in = config.D_in
    self.H = config.H
    self.W = config.W
    self.h_dim = config.h_dim
    self.patch_weight = config.patch_weight
    self.view_proc = config.view_proc

    self.reuse_resnet = False
    self.reuse_vgg = False
    self.reuse_mobilenet = False
    self.reuse_inception = False


    # regularization weights
    self.beta = 0.0001

    # input layers
    with tf.device( '/device:GPU:%i' % ( self.config.input_GPU ) ):
      with tf.variable_scope( 'input' ):

        self.lf = tf.placeholder(tf.float32, shape=[None, self.D_in*self.D_in, self.H, self.W, self.C])
        self.batch_size = tf.shape(self.lf)[0]
        # self.hs = tf.fill([self.batch_size,self.H,self.W,self.h_dim], 0.1)
        self.hs = tf.fill([self.batch_size, 3, 3, self.h_dim], 0.1)
        self.gen_pos = np.zeros([self.D_in * self.D_in - self.D_in * 2 + 1], dtype=int)

        self.lf_shape = self.lf.shape.as_list()
        self.lf_shape[0] = -1

        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)
        self.noise_sigma = tf.placeholder(tf.float32)

    # FEATURE LAYERS

    self.minimizers = dict()

    self.encoder_variables = dict()
    self.skip_variables = dict()
    self.decoder_variables = dict()
    self.LSTM_variables = dict()
    self.LFLSTM = dict()

    if self.view_proc == '3D':
        self.create_3D_encoders_variables()
    elif self.view_proc == '2D':
        self.create_2D_encoders_variables()
    self.create_2D_skip_variables()

    self.create_LSTM_variables()
    self.create_2D_decoders_variables()

    if len(config.discriminator) > 0:
        self.discriminator_config = config.discriminator[0]
        self.create_discriminator_variables()
        self.use_gan = True
    else:
        self.use_gan = False

    self.create_LFLSTM()

    if  self.use_gan == True:
        self.create_discriminator()

    self.setup_losses()

#
   # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
   #
  def create_3D_encoders_variables(self):
      # for encoder_config in self.config.encoders_3D:
          with tf.device('/device:GPU:%i' % (self.config.encoders_3D[0]['preferred_gpu'])):
              # self.create_3D_encoder(encoder_config)
              encoder_id = self.config.encoders_3D[0]['id']
              encoder = dict()
              layout = []
              for i in range(0, len(self.config.layers['encoder_3D'])):
                  layout.append(self.config.layers['encoder_3D'][i])
              with tf.variable_scope(encoder_id, reuse = tf.AUTO_REUSE):
                  encoder['variables'] = []
                  last_layer = min(len(layout), self.max_layer)
                  for i in range(0, last_layer):
                      layer_id = "encoder_%i" % i
                      print('    creating 3D encoder variables ' + layer_id)
                      encoder['variables'].append(layers.encoder_variables(layer_id, layout[i]))

                  self.encoder_variables = encoder['variables']

  def create_2D_encoders_variables(self):
      # for encoder_config in self.config.encoders_3D:
          with tf.device('/device:GPU:%i' % (self.config.encoders_3D[0]['preferred_gpu'])):
              # self.create_3D_encoder(encoder_config)
              encoder_id = self.config.encoders_3D[0]['id']
              encoder = dict()
              layout = []
              for i in range(0, len(self.config.layers['encoder_2D'])):
                  layout.append(self.config.layers['encoder_2D'][i])
              with tf.variable_scope(encoder_id, reuse = tf.AUTO_REUSE):
                  encoder['variables'] = []
                  last_layer = min(len(layout), self.max_layer)
                  for i in range(0, last_layer):
                      layer_id = "encoder_%i" % i
                      print('    creating 2D encoder variables ' + layer_id)
                      encoder['variables'].append(layers.encoder_variables_2D(layer_id, layout[i]))

                  self.encoder_variables = encoder['variables']

  def create_2D_skip_variables(self):
      # for encoder_config in self.config.encoders_3D:
          with tf.device('/device:GPU:%i' % (self.config.encoders_3D[0]['preferred_gpu'])):
              # self.create_3D_encoder(encoder_config)
              encoder_id = self.config.encoders_3D[0]['id']
              skip = dict()
              layout = []
              for i in range(0, len(self.config.layers['skip_2D'])):
                  layout.append(self.config.layers['skip_2D'][i])
              with tf.variable_scope(encoder_id, reuse = tf.AUTO_REUSE):
                  skip['variables'] = []
                  last_layer = min(len(layout), self.max_layer)
                  for i in range(0, last_layer):
                      layer_id = "skip_%i" % i
                      print('    creating 2D skip variables ' + layer_id)
                      skip['variables'].append(layers.encoder_variables_2D(layer_id, layout[i]))

                  self.skip_variables = skip['variables']

  def create_2D_decoders_variables( self ):
    # for decoder_config in self.config.decoders_3D:
        with tf.device('/device:GPU:%i' % (self.config.decoders_3D[0]['preferred_gpu'])):
            # self.create_3D_decoder( decoder_config)
            decoder = dict()
            decoder_id = self.config.decoders_3D[0]['id']
            decoder['variables'] = []
            layout = []
            for i in range(0, len(self.config.layers['decoder_2D'])):
                layout.append(self.config.layers['decoder_2D'][i])
            last_layer = min(len(layout), self.max_layer)
            with tf.variable_scope(decoder_id, reuse = tf.AUTO_REUSE):
                for i in range(0, last_layer):
                    layer_id = "decoder_%s_%i" % (decoder_id, i)
                    print('    generating upconvolution variables ' + layer_id)
                    # decoder['variables'].append(layers.encoder_variables_2D(layer_id, layout[i]))
                    decoder['variables'].append(layers.decoder_variables(layer_id, layout[i],i,last_layer,self.patch_weight))

                self.decoder_variables = decoder['variables']


  def create_LSTM_variables( self ):
    # for encoder_config in self.config.encoders_3D:
        with tf.device('/device:GPU:%i' % (self.config.encoders_3D[0]['preferred_gpu'])):
            LSTM_variables = dict()

            LSTM_variables['z_gate']=[]
            LSTM_variables['r_gate']=[]
            LSTM_variables['h_gate']=[]
            layout = []
            for i in range(0, len(self.config.layers['LSTM'])):
                layout.append(self.config.layers['LSTM'][i])
            last_layer = min(len(layout), self.max_layer)
            with tf.variable_scope('LSTM',reuse = tf.AUTO_REUSE):
                for i in range(0, last_layer):
                    layer_id = "Z_gate_%i"%i
                    print('    generating LSTM internal variables ' + layer_id)
                    LSTM_variables['z_gate'].append(layers.encoder_variables_2D(layer_id, layout[i]))
                    layer_id = "R_gate_%i"%i
                    print('    generating LSTM internal variables ' + layer_id)
                    LSTM_variables['r_gate'].append(layers.encoder_variables_2D(layer_id, layout[i]))
                    layer_id = "H_gate_%i"%i
                    print('    generating LSTM internal variables ' + layer_id)
                    LSTM_variables['h_gate'].append(layers.encoder_variables_2D(layer_id, layout[i]))

            self.LSTM_variables = LSTM_variables


  def create_discriminator_variables(self):
      with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
          with tf.variable_scope('discriminator', reuse = tf.AUTO_REUSE):
              discriminator = dict()
              discriminator['variables'] = []

              layout = []
              for i in range(0, len(self.config.layers['GAN_2D'])):
                  layout.append(self.config.layers['GAN_2D'][i])

              layout[0]['conv'][-2] = self.discriminator_config['channels'] * 3

              # create encoder variables
              last_layer = min(len(layout), self.max_layer)

              for i in range(0, last_layer):
                  layer_id = "discriminator_%i" % i
                  print('    creating GAN variables ' + layer_id)
                  discriminator['variables'].append(layers.discriminator_variables_2D(layer_id, layout[i]))

              self.discriminator_variables = discriminator['variables']


  def create_LFLSTM(self):
      with tf.device('/device:GPU:%i' % (self.config.LSTM_GPU)):
          print("Initializing hidden states and input images")
          LFLSTM = dict()
          LFLSTM['gen_views'] = []
          LFLSTM['features_dx'] = []
          LFLSTM['features_dy'] = []
          LFLSTM['features_gt_dx'] = []
          LFLSTM['features_gt_dy'] = []
          LFLSTM['losses_iter'] = 0
          LFLSTM['loss_fn'] = self.config.decoders_3D[0]['loss_fn']
          LFLSTM['train'] = self.config.decoders_3D[0]['train']
          LFLSTM['preferred_gpu'] = self.config.LSTM_GPU

          LFLSTM['input_LF'] = tf.placeholder(tf.float32, shape=[None, self.D_in*self.D_in, self.H, self.W, self.C])
          LFLSTM['GT'] = []

          all_states = []
          all_views = []
          for k in range(0,self.D_in**2):
            all_states.append(self.hs)
            all_views.append(self.lf[:,k,:,:,:])

          print("Done.")
          pos = 0
          for act_pos in range((self.D_in+1),self.D_in**2):
              # view generation row first
              if np.mod(act_pos, self.D_in) != 0: # skip existing input views
                  if self.view_proc == '3D':
                      img_input_1 = tf.expand_dims(all_views[act_pos - self.D_in-1],1)
                      img_input_2 = tf.expand_dims(all_views[act_pos - self.D_in],1)
                      img_input_3 = tf.expand_dims(all_views[act_pos - 1],1)
                      img_input = tf.concat([img_input_1,img_input_2,img_input_3],axis = 1)
                  elif self.view_proc == '2D':
                      img_input_1 = all_views[act_pos - self.D_in - 1]
                      img_input_2 = all_views[act_pos - self.D_in]
                      img_input_3 = all_views[act_pos - 1]
                      img_input = tf.concat([img_input_1, img_input_2, img_input_3], axis=-1)

                  state_h = all_states[act_pos - 1]
                  state_v = all_states[act_pos - self.D_in]

                  skip_features,features = ops.image_to_feature_downconv(img_input, self.encoder_variables,
                                                  self.skip_variables, self.phase,self.config.training)

                  new_state = ops.LSTM_Cell(features, state_v, state_h,
                                                      self.LSTM_variables['z_gate'],
                                                      self.LSTM_variables['r_gate'],
                                                      self.LSTM_variables['h_gate'],
                                                      self.phase, self.config.training)

                  new_view = ops.feature_to_image_upconv(new_state, skip_features,
                                                                    self.decoder_variables,
                                                                    self.batch_size,
                                                                    self.phase, self.config.training)

                  print('generating view %i' % act_pos)
                  self.gen_pos[pos] = act_pos
                  LFLSTM['gen_views'].append(new_view)

                  all_views[act_pos]= new_view
                  all_states[act_pos]= new_state

                  LFLSTM['GT'].append(LFLSTM['input_LF'][:,act_pos,:,:,:])
                  LFLSTM['features_dx'].append(tf.pad(new_view, [[0, 0], [0, 0], [0, 1], [0, 0]])[:, :, 1:, :] - new_view)
                  LFLSTM['features_dy'].append(tf.pad(new_view, [[0, 0], [0, 1], [0, 0], [0, 0]])[:, 1:, :, :] - new_view)
                  LFLSTM['features_gt_dx'].append(tf.pad(LFLSTM['input_LF'][:,act_pos,:,:,:], [[0, 0], [0, 0], [0, 1], [0, 0]])[:,
                                             :, 1:,:] - LFLSTM['input_LF'][:,act_pos,:,:,:])
                  LFLSTM['features_gt_dy'].append(tf.pad(LFLSTM['input_LF'][:,act_pos,:,:,:], [[0, 0], [0, 1], [0, 0], [0, 0]])[:,
                                             1:,:, :] - LFLSTM['input_LF'][:,act_pos,:,:,:])
                  pos = pos + 1

          LFLSTM['gen_views'] = tf.transpose(tf.convert_to_tensor(LFLSTM['gen_views']), perm=[1,0,2,3,4])
          LFLSTM['GT'] = tf.transpose(tf.convert_to_tensor(LFLSTM['GT']), perm=[1, 0, 2, 3, 4])

          self.LFLSTM = LFLSTM

  def inception_forward(self, x, layer, scope):
      mean = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
      x = tf.cast(x, tf.float32) * 2. - 1 + mean
      # send through resnet
      with slim.arg_scope(inception_v3_arg_scope()):
          _, layers = inception_v3(x, num_classes=None, is_training=False, reuse=self.reuse_inception)
      self.reuse_inception = True

      return layers[layer]

  def create_discriminator(self):
      with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):

              discriminator = dict()

              discriminator['features']=[]
              discriminator['features_gt']=[]
              discriminator['percep_gen']=[]
              discriminator['percep_gt']=[]

              layout = []
              for i in range(0, len(self.config.layers['GAN_2D'])):
                  layout.append(self.config.layers['GAN_2D'][i])
              # create encoder variables
              last_layer = min(len(layout), self.max_layer)

              no_relu = False

              for view in range(0,len(self.gen_pos)):
                  features = tf.concat(
                      [self.LFLSTM['gen_views'][:,view, 0:-1, 0:-1, :],
                       self.LFLSTM['features_dx'][view][:, 0:-1, 0:-1, :],
                       self.LFLSTM['features_dy'][view][:, 0:-1, 0:-1, :]], axis=-1)

                  features_gt = tf.concat(
                      [self.LFLSTM['GT'][:,view, 0:-1, 0:-1, :],
                       self.LFLSTM['features_gt_dx'][view][:, 0:-1, 0:-1, :],
                       self.LFLSTM['features_gt_dy'][view][:, 0:-1, 0:-1, :]], axis=-1)


                  for i in range(0, last_layer):
                      layer_id_v = "v_%s_%i" % ('discriminator_', i)
                      print('    generating downconvolution layer structure for %s %i' % ('discriminator', i))
                      if i == last_layer - 1:
                          no_relu = True
                      features = layers.discriminator_conv2d(layer_id_v,
                                                             self.discriminator_variables[i],
                                                             features,
                                                             no_relu=no_relu).out

                      features_gt = layers.discriminator_conv2d(layer_id_v + '_gt',
                                                                self.discriminator_variables[i],
                                                                features_gt,
                                                                no_relu=no_relu).out

                      if i == self.discriminator_config['percep_layer']-1:
                          discriminator['percep_gen'].append(features)
                          discriminator['percep_gt'].append(features_gt)

                  discriminator['features'].append(features)
                  discriminator['features_gt'].append(features_gt)

              discriminator['features'] = tf.transpose(tf.convert_to_tensor(discriminator['features']),
                                                       perm=[1, 0, 2, 3, 4])
              discriminator['features_gt'] = tf.transpose(tf.convert_to_tensor(discriminator['features_gt']),
                                                          perm=[1, 0, 2, 3, 4])
              discriminator['percep_gen'] = tf.transpose(tf.convert_to_tensor(discriminator['percep_gen']),
                                                       perm=[1, 0, 2, 3, 4])
              discriminator['percep_gt'] = tf.transpose(tf.convert_to_tensor(discriminator['percep_gt']),
                                                          perm=[1, 0, 2, 3, 4])



              # create dense layers
              print('    creating dense layers for discriminator')
              sh = discriminator['features'].shape.as_list()
              discriminator['encoder_input_size'] = sh[1] * sh[2] * sh[3] * sh[4]
              # setup shared feature space between horizontal/vertical encoder
              discriminator['features_transposed'] = tf.reshape(discriminator['features'],[-1, discriminator['encoder_input_size']])
              discriminator['features_transposed_gt'] = tf.reshape(discriminator['features_gt'],[-1, discriminator['encoder_input_size']])

              discriminator['discriminator_nodes'] = discriminator['features_transposed'].shape.as_list()[1]

              discriminator['W_dense'], discriminator['b_dense'] = layers.bn_dense_discriminator_variables(discriminator['discriminator_nodes'], 1, 'bn_gan_out')

              discriminator['logits'] = layers.bn_dense_discriminator(discriminator['features_transposed'],
                                                                      discriminator['W_dense'],
                                                                      discriminator['b_dense'])

              discriminator['logits_gt'] = layers.bn_dense_discriminator(discriminator['features_transposed_gt'],
                                                                         discriminator['W_dense'],
                                                                         discriminator['b_dense'])

              self.discriminator = discriminator



  def add_training_ops( self ):

    print( 'creating training ops' )

    # what needs to be updated before training
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # L2-loss on feature layers

    for cfg in self.config.minimizers:
        if 'losses_3D' in cfg:
            # counter = 0
            # for scale in self.scales:
                minimizer = dict()
                minimizer['id'] = cfg['id']
                print('  minimizer ' + minimizer['id'])

                with tf.device('/device:GPU:%i' % (cfg['preferred_gpu'])):
                    # counter +=1
                    minimizer['loss'] = 0
                    minimizer['requires'] = []
                    # minimizer['requires'].append(scale)

                    if self.LFLSTM['train']:
                        minimizer['loss'] += self.config.L2_weight * self.LFLSTM['losses'] + self.LFLSTM['diff_losses'] + \
                                             self.config.adv_weight * self.LFLSTM['loss_adv']
                    with tf.control_dependencies( self.update_ops ):
                        # Ensures that we execute the update_ops before performing the train_step
                        minimizer['orig_optimizer'] = tf.train.AdamOptimizer(cfg['step_size'])
                        minimizer['optimizer'] = tf.contrib.estimator.clip_gradients_by_norm(
                            minimizer['orig_optimizer'],
                            clip_norm=5.0)
                        minimizer['train_step'] = minimizer['optimizer'].minimize(minimizer['loss'],
                                                                                  var_list=[v for v in
                                                                                            tf.global_variables() if
                                                                                            "discriminator" not in v.name])
                    self.minimizers[ cfg[ 'id' ] ] = minimizer

    if self.use_gan:
        minimizer = dict()
        minimizer['loss'] = 0
        minimizer['requires'] = []

        minimizer['id'] = self.GAN_loss['id']
        print('  minimizer ' + minimizer['id'])
        with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
            minimizer['loss'] += self.discriminator_config['weight'] * self.GAN_loss['loss']
            # clip weights in D
            # clip_values = [-0.01, 0.01]  # [-0.01, 0.01]
            var_list_grid = [v for v in tf.global_variables() if "discriminator" in v.name]
            # self.clip_discriminator_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var
            #                               in var_list_grid]
        with tf.control_dependencies(self.update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            minimizer['orig_optimizer'] = tf.train.AdamOptimizer(self.discriminator_config['step_size'])
            minimizer['optimizer'] = tf.contrib.estimator.clip_gradients_by_norm(minimizer['orig_optimizer'],
                                                                                 clip_norm=5.0)
            minimizer['train_step'] = minimizer['optimizer'].minimize(minimizer['loss'],
                                                                      var_list=var_list_grid)
        self.minimizers[minimizer['id']] = minimizer

  # add training ops for additional decoder pathway (L2 loss)
  def setup_losses( self ):
   loss_summary = dict()

   # for id in self.decoders_3D:
       # loss function for auto-encoder
   with tf.variable_scope('training_LFLSTM'):
       if self.LFLSTM['loss_fn'] == 'L2':
           # counter = 0
           # for scale in self.scales:
           with tf.device('/device:GPU:%i' % (self.LFLSTM['preferred_gpu'])):
               # counter +=1
               print('  creating L2-loss')
               self.LFLSTM['losses'] = 0
               self.LFLSTM['losses'] = tf.losses.mean_squared_error(self.LFLSTM['GT'][:,:,:,:,0],self.LFLSTM['gen_views'][:,:,:,:,0]) \
                                       + tf.losses.mean_squared_error(self.LFLSTM['GT'][:,:,:,:,1],self.LFLSTM['gen_views'][:,:,:,:,1]) \
                                       + tf.losses.mean_squared_error(self.LFLSTM['GT'][:,:,:,:,2],self.LFLSTM['gen_views'][:,:,:,:,2])
               self.LFLSTM['diff_losses'] = 0
               self.LFLSTM['diff_losses'] = tf.losses.mean_squared_error(self.LFLSTM['features_gt_dy'],self.LFLSTM['features_dy']) + \
                                            tf.losses.mean_squared_error(self.LFLSTM['features_gt_dx'],self.LFLSTM['features_dx'])

               loss_summary['LFLSTM'] = tf.summary.scalar('loss_LFLSTM', self.LFLSTM['losses'])
               loss_summary['LFLSTM_diff'] = tf.summary.scalar('diff_loss_LFLSTM', self.LFLSTM['diff_losses'])

       # if self.use_gan:
       #     with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
       #         with tf.variable_scope('discriminator'):
       #             GAN_loss = dict()
       #             GAN_loss['id'] = 'GAN'
       #             GAN_loss['loss'] = 0
       #             errD = tf.reduce_mean(self.discriminator['features_gt'] - self.discriminator['features'])
       #             GAN_loss['loss'] += errD
       #         self.GAN_loss = GAN_loss
       #         loss_summary['GAN'] = tf.summary.scalar('loss_GAN', self.GAN_loss['loss'])
       #
       # if self.use_gan:
       #     counter = 0
       #     with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
       #         counter += 1
       #         self.LFLSTM['loss_adv'] = tf.reduce_mean(self.discriminator['features'])
       #         loss_summary['GAN_ADV'] = tf.summary.scalar('loss_adv', self.LFLSTM['loss_adv'])


       if self.use_gan:
           with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
               with tf.variable_scope('discriminator'):
                   GAN_loss = dict()
                   GAN_loss['id'] = 'GAN'
                   GAN_loss['loss'] = 0

                   loss_real = tf.reduce_mean(
                       tf.losses.sigmoid_cross_entropy(tf.ones_like(self.discriminator['logits_gt']),
                                                       self.discriminator['logits_gt']))
                   loss_fake = tf.reduce_mean(
                       tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.discriminator['logits']),
                                                       self.discriminator['logits']))
                   GAN_loss['loss'] += (loss_real + loss_fake) / 2
                   self.GAN_loss = GAN_loss
                   loss_summary['GAN'] = tf.summary.scalar('loss_GAN', self.GAN_loss['loss'])

       if self.use_gan:
           counter = 0
           with tf.device('/device:GPU:%i' % (self.config.GAN_GPU)):
               counter += 1
               self.LFLSTM['loss_percep_adv'] = tf.losses.mean_squared_error(self.discriminator['percep_gt'],self.discriminator['percep_gen'])
               self.LFLSTM['loss_adv'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                   logits=self.discriminator['logits'],
                   labels=tf.ones_like(self.discriminator['logits']))) + \
                    self.discriminator_config['percep_weight']*self.LFLSTM['loss_percep_adv']

               loss_summary['GAN_ADV'] = tf.summary.scalar('loss_adv', self.LFLSTM['loss_adv'])

   image_summary = dict()
   if self.config.config['ColorSpace'] == 'RGB':

           image_summary['lf_res'] = tf.summary.image('lf_res', tf.clip_by_value(
               self.LFLSTM['gen_views'][:,self.D_in**2-self.D_in*2,:,:,:], 0.0, 1.0), max_outputs=3)

           image_summary['lf_input'] = tf.summary.image('lf_input',
               self.LFLSTM['GT'][:,self.D_in**2-self.D_in*2,:,:,:], max_outputs = 3)

   self.merged_images = tf.summary.merge([ v for k,v in image_summary.items()])
   self.merged_lstm = tf.summary.merge([v for k, v in loss_summary.items()])
   if self.use_gan:
       self.merged_gan = tf.summary.merge([ v for k,v in loss_summary.items() if k.startswith('GAN')])

  # initialize new variables
  def initialize_uninitialized( self, sess ):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
    for i in not_initialized_vars:
      print( str(i.name) )

    if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))

  # prepare input
  def prepare_net_input( self, batch ):
      net_in = {  self.keep_prob:       1.0,
                  self.phase:           True,
                  self.noise_sigma:     self.config.training[ 'noise_sigma' ] }


      net_in[self.lf] = batch['lf_patches'][:,0:self.D_in,0:self.D_in,:,:,:].reshape(
                                   [-1, self.D_in*self.D_in, self.H, self.W, self.C])

      net_in[self.LFLSTM['input_LF']] = batch['lf_patches'][:,0:self.D_in,0:self.D_in,:,:,:].reshape(
                                   [-1, self.D_in*self.D_in, self.H, self.W, self.C])

      return net_in
