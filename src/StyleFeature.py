from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import save_img, get_img, exists, list_files
import cv2

vgg_path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/GAN-UNet/data/imagenet-vgg-verydeep-19.mat'

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = ('relu4_2',)


'''
###get the style features of style images
'''
def get_style_features(style_img):

    image = tf.multiply(style_img + 1, 127.5)

    if image._shape_as_list()[1] != 128:
        image = tf.image.resize_images(image,[128, 128])

    style_features = {}

    style_image_pre = vgg.preprocess(image)
    net = vgg.net(vgg_path, style_image_pre)

    for layer in STYLE_LAYERS:
        features = net[layer]       #.eval(feed_dict={style_img: style_image_pre})
        features = tf.reshape(features, shape=[-1, features._shape_as_list()[1]*features._shape_as_list()[2], features._shape_as_list()[3]])[0]
        gram = tf.matmul(tf.transpose(features), features) / float(features._shape_as_list()[0]*features._shape_as_list()[1])
        style_features[layer] = gram

    return style_features


'''
### get the content features of content images
'''
def get_content_features(content_img):

    image = tf.multiply(content_img + 1, 127.5)

    if image._shape_as_list()[1] != 128:
        image = tf.image.resize_images(image, [128, 128])

    content_features = {}

    X_pre = vgg.preprocess(image)
    content_net = vgg.net(vgg_path, X_pre)

    for layer in CONTENT_LAYER:
        content_features[layer] = content_net[layer]    #.eval(feed_dict={content_img: X_pre})

    return content_features


'''
### get the style loss by calculating the difference 
between the style_image and genetated image
'''
def get_style_loss(style_features, style_img):

    style_img_features = get_style_features(style_img)

    style_lossE = 0
    for style_layer in STYLE_LAYERS:
        coff = float(1.0 / len(STYLE_LAYERS))
        img_gram = style_img_features[style_layer]
        style_gram = style_features[style_layer]
        style_lossE += coff * tf.reduce_mean(tf.abs(img_gram - style_gram))


    style_loss = tf.reduce_mean(style_lossE)

    return style_loss


'''
### get the content loss by calculating the difference 
    between the synthesized image and genetated image
'''
def get_content_loss(img, syn_img):

    img_features = get_content_features(img)
    syn_features = get_content_features(syn_img)

    content_lossE = 0
    for content_layer in CONTENT_LAYER:
        coff = float(1.0 / len(CONTENT_LAYER))
        img_content = img_features[content_layer]
        syn_content = syn_features[content_layer]
        content_lossE += coff * tf.reduce_mean(tf.abs(img_content - syn_content))

    content_loss = tf.reduce_mean(content_lossE)

    return content_loss


'''
### get the total variation loss
'''
def get_tv_loss(preds):

    img = preds

    tv_loss = tf.reduce_mean(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :])) + \
              tf.reduce_mean(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    return tv_loss






























