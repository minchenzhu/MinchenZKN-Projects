#
# A bunch of useful helper functions to work with
# the light field data.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from libs.convert_colorspace import rgb2YCbCr, rgb2YUV, YCbCr2rgb
import cv2
import st
from scipy.interpolate import interpn


# returns two epipolar plane image stacks (horizontal/vertical),
# block size (xs,ys), block location (x,y), both in pixels.
def epi_stacks(LF, y, x, ys, xs):
    T = np.int32(LF.shape[0])
    cv_v = np.int32((T - 1) / 2)
    S = np.int32(LF.shape[1])
    cv_h = np.int32((S - 1) / 2)
    stack_h = LF[cv_v, :, y:y + ys, x:x + xs, :]
    stack_v = LF[:, cv_h, y:y + ys, x:x + xs, :]
    return (stack_h, stack_v)


def epi_stacks_2(LF, y, x, ys, xs):
    T = np.int32(LF.shape[0])
    cv_v = np.int32((T - 1) / 2)
    S = np.int32(LF.shape[1])
    cv_h = np.int32((S - 1) / 2)
    stack_h = LF[cv_v, :, y:y + ys, x:x + xs, :]
    stack_v = LF[:, cv_h, y:y + ys, x:x + xs, :]

    order1 = np.arange(T)
    order2 = sorted(order1, reverse=True)
    stack_l = LF[order1, order1, y:y + ys, x:x + xs, :]
    stack_r = LF[order1, order2, y:y + ys, x:x + xs, :]

    return (stack_h, stack_v, stack_l, stack_r)


# returns center view
def cv(LF):
    T = np.int32(LF.shape[0])
    cv_v = np.int32((T - 1) / 2)
    S = np.int32(LF.shape[1])
    cv_h = np.int32((S - 1) / 2)
    return LF[cv_v, cv_h, :, :, :]


# show an image (with a bunch of checks)
def show(img, cmap='gray'):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    if img.shape[2] == 1:
        img = img[:, :, 0]
        # img = np.clip( img, 0.0, 1.0 )
        imgplot = plt.imshow(img, interpolation='none', cmap=cmap)
    else:
        imgplot = plt.imshow(img, interpolation='none')
    plt.show(block=False)


def augment_data(input, idx):
    size = input['stacks_v'][idx].shape
    a_rdm = np.random.randn(1, 1, 1, 3) / 16 + 1
    b_rdm = np.random.randn(1, 1, 1, 3) / 16
    size = input['stacks_v_HR'][idx].shape
    a_HR = np.tile(a_rdm, (size[0], size[1], size[2], 1))
    b_HR = np.tile(b_rdm, (size[0], size[1], size[2], 1))

    stacks_v_HR = input['stacks_v_HR'][idx]
    stacks_h_HR = input['stacks_h_HR'][idx]
    stacks_v_HR = augment_albedo(a_HR, b_HR, stacks_v_HR)
    stacks_h_HR = augment_albedo(a_HR, b_HR, stacks_h_HR)

    input['stacks_v_HR'][idx] = stacks_v_HR
    input['stacks_h_HR'][idx] = stacks_h_HR

    return (input)


def augment_data_YCBCR(input, idx):
    a_rdm = np.random.randn(1, 1, 1, 3) / 16 + 1
    b_rdm = np.random.randn(1, 1, 1, 3) / 16
    size = input['stacks_v_HR'][idx].shape
    a_HR = np.tile(a_rdm, (size[0], size[1], size[2], 1))
    b_HR = np.tile(b_rdm, (size[0], size[1], size[2], 1))

    stacks_v_HR = input['stacks_v_HR'][idx]
    stacks_h_HR = input['stacks_h_HR'][idx]
    stacks_v_HR = augment_albedo_YCBCR(a_HR, b_HR, stacks_v_HR)
    stacks_h_HR = augment_albedo_YCBCR(a_HR, b_HR, stacks_h_HR)

    input['stacks_v_HR'][idx] = stacks_v_HR
    input['stacks_h_HR'][idx] = stacks_h_HR

    return (input)

def create_scales(input, idx):
    if 'stacks_v_s4' in input:
        input['stacks_v_s4'][idx] = np.clip(input['stacks_v_HR'][idx], 16.0 / 255.0, 240 / 255)
        input['stacks_h_s4'][idx] = np.clip(input['stacks_h_HR'][idx], 16.0 / 255.0, 240 / 255)

        input['stacks_v_s2'][idx] = np.clip(np.stack([cv2.resize(input['stacks_v_HR'][idx][i, :, :, :], (96,96),
                                                        interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
        input['stacks_h_s2'][idx] = np.clip(np.stack([cv2.resize(input['stacks_h_HR'][idx][i, :, :, :], (96,96),
                                                        interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
        input['stacks_v'][idx] = np.clip(np.stack([cv2.resize(input['stacks_v_HR'][idx][i, :, :, :], (48,48),
                                                        interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
        input['stacks_h'][idx] = np.clip(np.stack([cv2.resize(input['stacks_h_HR'][idx][i, :, :, :], (48,48),
                                                        interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
    else:
        input['stacks_v_s2'][idx] = np.clip(input['stacks_v_HR'][idx], 16.0 / 255.0, 240 / 255)
        input['stacks_h_s2'][idx] = np.clip(input['stacks_h_HR'][idx], 16.0 / 255.0, 240 / 255)
        input['stacks_v'][idx][...,0:3] = np.clip(np.stack([cv2.resize(input['stacks_v_HR'][idx][i, :, :, :], (48, 48),
                                                     interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)
        input['stacks_h'][idx][...,0:3] = np.clip(np.stack([cv2.resize(input['stacks_h_HR'][idx][i, :, :, :], (48, 48),
                                                     interpolation=cv2.INTER_CUBIC) for i in range(0, 9)]), 16.0 / 255.0, 240 / 255)

    # biclinear stuff
    [tq, vq, uq] = np.meshgrid(range(9), range(48), range(48))
    tq = tq.transpose(1, 0, 2)
    vq = vq.transpose(1, 0, 2)
    uq = uq.transpose(1, 0, 2)
    points = [[0,4,8], np.arange(48), np.arange(48)]
    xi =  (tq.ravel(), vq.ravel(), uq.ravel())
    for ch in range(0,3):
        input['stacks_bicubic_v'][idx][:,:,:,ch] = np.clip(interpn(points, input['stacks_v'][idx][0:9:4,:,:,ch], xi).reshape((9, 48, 48)), 16.0 / 255.0, 240 / 255)
        input['stacks_bicubic_h'][idx][:, :, :, ch] = np.clip(interpn(points, input['stacks_h'][idx][0:9:4, :, :, ch],
                                                              xi).reshape((9, 48, 48)), 16.0 / 255.0, 240 / 255)
    return (input)

def augment_data_Y(input, idx):
    a_rdm = np.random.randn(1, 1, 1, 3) / 8 + 1
    b_rdm = np.random.randn(1, 1, 1, 3) / 8
    size = input['stacks_v_HR'][idx].shape
    a_HR = np.tile(a_rdm, (size[0], size[1], size[2], 1))
    b_HR = np.tile(b_rdm, (size[0], size[1], size[2], 1))

    stacks_v_HR = input['stacks_v_HR'][idx]
    stacks_h_HR = input['stacks_h_HR'][idx]
    stacks_v_HR = augment_albedo_YCBCR(a_HR, b_HR, stacks_v_HR)
    stacks_h_HR = augment_albedo_YCBCR(a_HR, b_HR, stacks_h_HR)

    input['stacks_v_HR'][idx] = stacks_v_HR[...,0:1]
    input['stacks_h_HR'][idx] = stacks_h_HR[...,0:1]

    return (input)


def augment_data_HSV(input, idx):
    size_hs = input['stacks_v_hs'][idx].shape
    a = np.tile(np.random.randn(1, 1, 1, 2) / 8 + 1, (size_hs[0], size_hs[1], size_hs[2], 1))
    b = np.tile(np.random.randn(1, 1, 1, 2) / 8, (size_hs[0], size_hs[1], size_hs[2], 1))

    stacks_v_hs = input['stacks_v_hs'][idx]
    stacks_h_hs = input['stacks_h_hs'][idx]
    stacks_v_hs = augment_albedo(a, b, stacks_v_hs)
    stacks_h_hs = augment_albedo(a, b, stacks_h_hs)

    input['stacks_v_hs'][idx] = stacks_v_hs
    input['stacks_h_hs'][idx] = stacks_h_hs

    size_v = input['stacks_v_v'][idx].shape
    a = np.tile(np.random.randn(1, 1, 1, 1) / 8 + 1, (size_v[0], size_v[1], size_v[2], 1))
    b = np.tile(np.random.randn(1, 1, 1, 1) / 8, (size_v[0], size_v[1], size_v[2], 1))

    stacks_v_v = input['stacks_v_v'][idx]
    stacks_h_v = input['stacks_h_v'][idx]
    stacks_v_v = augment_albedo(a, b, stacks_v_v)
    stacks_h_v = augment_albedo(a, b, stacks_h_v)

    input['stacks_v_v'][idx] = stacks_v_v
    input['stacks_h_v'][idx] = stacks_h_v

    return (input)


def convert2YUV(input, stream, idx):
    streamtmp = input[stream][idx]

    streamtmp = rgb2YUV(streamtmp)

    input[stream][idx] = streamtmp

    return (input)


def convert2YCBCR(input, stream, idx):
    streamtmp = input[stream][idx]

    streamtmp = rgb2YCbCr(np.clip(streamtmp, 0.0, 1.0))

    input[stream][idx] = streamtmp

    return (input)


def convert2LAB(input, stream, idx):
    streamtmp = input[stream][idx]

    streamtmp = rgb2lab(streamtmp)

    input[stream][idx] = streamtmp

    return (input)


def augment_albedo(a, b, albedo):
    out = np.multiply(a, albedo) + b
    out = out - np.minimum(0, np.amin(out))
    out = np.clip(np.divide(out, np.maximum(np.amax(out), 1)),0.0,1.0)
    return (out)


def augment_albedo_YCBCR(a, b, albedo):
    out = np.multiply(a, albedo) + b
    out = out - np.minimum(0, np.amin(out))
    out = np.clip(np.divide(out, np.maximum(np.amax(out), 1)), 16.0 / 255.0, 240.0 / 255.0)
    tmp = out[..., 0:1]
    tmp = np.clip(tmp, 16.0 / 255.0, 235.0 / 255.0)
    out[..., 0:1] = tmp

    return (out)


def augment_data_intrinsic(input, idx):
    size = input['stacks_v'][idx].shape

    d = np.tile(np.abs(np.random.randn(1, 1, 1, 3) / 8 + 1), (size[0], size[1], size[2], 1))
    c = np.abs(np.random.randn(1) / 4 + 1)
    diffuse_v = input['diffuse_v'][idx]
    diffuse_h = input['diffuse_h'][idx]

    input['diffuse_v'][idx] = np.multiply(diffuse_v, d)
    input['diffuse_h'][idx] = np.multiply(diffuse_h, d)

    specular_v = input['specular_v'][idx]
    specular_h = input['specular_h'][idx]

    specular_v = np.multiply(c, specular_v)
    specular_h = np.multiply(c, specular_h)

    input['specular_v'][idx] = specular_v
    input['specular_h'][idx] = specular_h

    input['stacks_v'][idx] = input['diffuse_v'][idx] + input['specular_v'][idx]
    input['stacks_h'][idx] = input['diffuse_h'][idx] + input['specular_h'][idx]

    return (input)


# visualize an element of a batch for training/test
def show_batch(batch, n):
    ctr = 4

    # vertical stack
    plt.subplot(2, 2, 1)
    plt.imshow(batch['stacks_v'][n, :, :, 24, :])

    # horizontal stack
    plt.subplot(2, 2, 2)
    plt.imshow(batch['stacks_h'][n, :, :, 24, :])

    # vertical stack center
    plt.subplot(2, 2, 3)
    plt.imshow(batch['stacks_v'][n, ctr, :, :, :])

    # horizontal stack center
    plt.subplot(2, 2, 4)
    plt.imshow(batch['stacks_h'][n, ctr, :, :, :])

    plt.show()
