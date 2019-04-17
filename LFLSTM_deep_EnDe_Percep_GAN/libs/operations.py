import tensorflow as tf
import numpy as np
import math
import libs.layers as layers
import tensorflow.contrib.slim as slim

def image_to_feature(view_proc, input, variables, phase, train_config):
    features = input
    if view_proc == '3D':
        for i in range(0, len(variables)):
            layer_id_v = "v_%s_%i" % ('img2feat', i)
            # print('    generating convolutional layer structure for %s %i' % ('img2feat', i))
            features = layers.layer_conv3d(layer_id_v,
                                           variables[i],
                                           features,
                                           phase,
                                           train_config).out
        features_cat = tf.concat([features[:, 0, :, :, :], features[:, 1, :, :, :], features[:, 2, :, :, :]],
                                 axis=-1)
    elif view_proc == '2D':
        for i in range(0, len(variables)):
            layer_id_v = "v_%s_%i" % ('img2feat', i)
            # print('    generating convolutional layer structure for %s %i' % ('img2feat', i))
            features = layers.layer_conv2d(layer_id_v,
                                           variables[i],
                                           features,
                                           phase,
                                           train_config).out

        features_cat = features

    return features_cat


def image_to_feature_downconv(input, variables, skip_variables, phase, train_config):
    features = []
    features.append(input)
    for i in range(0, len(variables)):
        layer_id_v = "v_%s_%i" % ('img2feat', i)
        # print('    generating convolutional layer structure for %s %i' % ('img2feat', i))
        features.append(layers.layer_conv2d(layer_id_v,
                                       variables[i],
                                       features[i],
                                       phase,
                                       train_config).out)

    features_cat = features[-1]
    skip = []
    for i in range(0, len(skip_variables)):
        layer_id_v = "v_%s_%i" % ('feat2skip', i)
        # print('    generating convolutional layer structure for %s %i' % ('img2feat', i))
        skip.append(layers.layer_conv2d(layer_id_v,
                                       skip_variables[i],
                                       features[-2-i],
                                       phase,
                                       train_config).out)
    return skip, features_cat




def LSTM_Cell(features, state_v, state_h, variables_Z, variables_R, variables_H, phase, train_config):

    # concat_state = tf.concat([state_v, state_h], axis = -1)
    concat_state = state_v + state_h
    concat_input = tf.concat([features, state_v, state_h], axis = -1)

    features_Z = concat_input
    features_R = concat_input

    for i in range(0, len(variables_Z)):
        layer_id_v = "v_%s_%i" % ('z_gate_feat', i)
        # print('    generating convolutional layer structure for %s %i' % ('feat2img', i))
        features_Z = layers.layer_conv2d(layer_id_v,
                                          variables_Z[i],
                                          features_Z,
                                          phase,
                                          train_config).out
    Feature_Z = tf.nn.sigmoid(features_Z)

    for i in range(0, len(variables_R)):
        layer_id_v = "v_%s_%i" % ('r_gate_feat', i)
        # print('    generating convolutional layer structure for %s %i' % ('feat2img', i))
        features_R = layers.layer_conv2d(layer_id_v,
                                          variables_R[i],
                                          features_R,
                                          phase,
                                          train_config).out
    Feature_R = tf.nn.sigmoid(features_R)

    features_H = tf.concat([features,tf.multiply(Feature_R, state_v),tf.multiply(Feature_R, state_h)],axis = -1)
    for i in range(0, len(variables_H)):
        layer_id_v = "v_%s_%i" % ('h_gate_feat', i)
        # print('    generating convolutional layer structure for %s %i' % ('feat2img', i))
        features_H = layers.layer_conv2d(layer_id_v,
                                          variables_H[i],
                                          features_H,
                                          phase,
                                          train_config).out
    Feature_H = tf.nn.tanh(features_H)

    new_state = tf.multiply(Feature_Z, concat_state) + tf.multiply((1-Feature_Z),Feature_H)

    # new_feature = tf.concat([features, state_v, state_h], axis = -1)

    return new_state



def feature_to_image(features, variables, phase, train_config):

    output = features
    for i in range(0, len(variables)):
        layer_id_v = "v_%s_%i" % ('feat2img', i)
        output = layers.layer_conv2d(layer_id_v,
                                    variables[i],
                                    output,
                                    phase,
                                    train_config).out

    return output

def feature_to_image_upconv(features, skip_features, variables, batch_size, phase, train_config):

    output = features
    for i in range(0, len(variables)):
        layer_id_v = "v_%s_%i" % ('feat2img', i)
        sh = output.get_shape().as_list()
        output_shape = [sh[1]*variables[i].stride[1],
                        sh[2]*variables[i].stride[2]]
        if i < len(skip_features):
            output = layers.layer_upconv2d_v2(layer_id_v,
                                                   variables[i], batch_size,
                                                   tf.concat([output,skip_features[i]],axis = -1),
                                                   phase,
                                                   train_config).out
        else:
            output = layers.layer_upconv2d_v2(layer_id_v,
                                                   variables[i],batch_size,
                                                   output,
                                                   phase,
                                                   train_config).out

    return output


