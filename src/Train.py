#keep compatability among different python version
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time
import os
import Generator
import Unet
import GAN
import StyleFeature
import vgg

content_weight = 2.0
style_weight = 0.15
tv_weight = 0.5

epoches = 100
batch_size = 16

#calculate training time
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

#Define our input and output data
#load all augmented depth images and labels
train_images = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Image.npy')
train_images = train_images.reshape([-1,128,128])
print('train_images.shape: ', train_images.shape)

train_labels = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Label.npy')
train_labels = train_labels.reshape([-1,42])
print('train_labels.shape: ', train_labels.shape)

tf.reset_default_graph()

# input depth image and hand joints
X_in_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='X_in_image')
X_in_label = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='X_in_label')

# inject the style image to vgg net
style_img = tf.placeholder(tf.float32, shape=[None,128,128,1], name="style_img")


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
vgg_path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/GAN-UNet/data/imagenet-vgg-verydeep-19.mat'

ss = np.load('myStyle_25.npy')
ss = (ss[0] + 1) * 127.5
style_batch = np.zeros([batch_size,128,128])

for i, single in enumerate(style_batch):
    style_batch[i] = ss

ss = np.reshape(ss, [-1,128,128,1])


style_features = {}

# precompute style features
with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:

    style_image = tf.placeholder(tf.float32, shape=[None,128,128,1], name='style_image')
    style_image_pre = vgg.preprocess(style_image)
    net = vgg.net(vgg_path, style_image_pre)

    for layer in STYLE_LAYERS:
        features = net[layer].eval(feed_dict={style_image: ss})
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        style_features[layer] = gram


'''
stage 1: generate depth images from joints distribution
'''
gen = Generator
img_gen = gen.generator(X_in_label)
img_gen_trans = tf.reshape(img_gen, [-1,128,128])
# loss_generator = tf.reduce_mean(tf.abs(img_gen_trans - X_in_image))
loss_generator = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(img_gen_trans, X_in_image), 1))

'''
stage 2: keep the more details through U-net.
'''
# unet = Unet
# img_unet = unet.unet(img_gen)
# img_unet_trans = tf.reshape(img_unet, [-1,128,128])
# loss_gen_unet = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(img_unet_trans, X_in_image), 1))

'''
stage 3: GAN
'''
gan = GAN
d_real, _, dis_hidden_layer = gan.discriminator(X_in_image)
d_fake, _, _ = gan.discriminator(img_gen, reuse=True)
dis_joints = gan.posterior_recognition(dis_hidden_layer)

loss_gan_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

loss_gan_joints = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(dis_joints, X_in_label), 1))


loss_gan_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

'''
stage4: calculate the losses of style transferring
'''
syn = img_gen
syn = tf.reshape(syn,[-1,128,128,1])
print("syn.shape",syn.shape)


styleT = StyleFeature
style_loss = styleT.get_style_loss(style_features, syn)

X_image = tf.reshape(X_in_image,[-1,128,128,1])
content_loss = styleT.get_content_loss(X_image, syn)

tv_loss = styleT.get_tv_loss(syn)

loss_style_transfer = content_weight*content_loss + style_weight*style_loss + tv_weight*tv_loss


'''
stage5: calculate the losses of the generator and discriminator
'''
loss_dis = loss_gan_dis + loss_gan_joints
loss_gen = loss_generator + loss_gan_gen + loss_style_transfer #+ loss_gen_unet  loss_generator +

# set RMSOptimizer to decrease the generator and discriminator losses
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):  #smooth_loss +

    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_dis)  ##+ loss_dis_joints + d_joints_reg

    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_gen)#

    #RMSPropOptimizer
'''
Training process
'''
steps = []
gen_loss_list = []
dis_loss_list = []

path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/GAN-UNet/results/'

'''
Start session and initialize all the variables
'''

init = tf.global_variables_initializer()

saver = tf.train.Saver()

start_time_sum = time.time()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(epoches):

        idx = np.random.randint(0, train_labels.shape[0], train_labels.shape[0])
        images = train_images[idx]
        labels = train_labels[idx]

        for step in range(0, train_labels.shape[0] // batch_size):

            # input depth image to GAN
            batch_image = images[step * batch_size : (step + 1) * batch_size, ]
            batch_joint = labels[step * batch_size : (step + 1) * batch_size, ]


            g_ls, d_ls = sess.run([loss_gen, loss_dis],
                                  feed_dict={X_in_image: batch_image, X_in_label: batch_joint, style_img: ss})  #, Noise: batch_noise

            start_time = time.time()

            sess.run(optimizer_d, feed_dict={X_in_image: batch_image, X_in_label: batch_joint, style_img: ss})

            sess.run(optimizer_g, feed_dict={X_in_image: batch_image, X_in_label: batch_joint, style_img: ss})

            duration = time.time() - start_time

            print("Epoch: %d, Step: %d,  Dis_loss: %f, Gen_loss: %f, Duration:%f sec"
                  % (epoch, step, d_ls, g_ls, duration))

            #save the loss of generator and discriminator
            steps.append(epoch * train_labels.shape[0] // batch_size + step)
            gen_loss_list.append(g_ls)
            dis_loss_list.append(d_ls)

            #show the last batch of images
            if not step % 200:

                gen_img = sess.run(img_gen, feed_dict={X_in_label: batch_joint, style_img: ss})
                gen_img = gen_img.reshape([-1,128,128])

                r, c = 2, 2
                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(gen_img[cnt, :], cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1
                fig.savefig(os.path.join(path + 'gen_images/', '%d_%d.png' % (epoch, step)))
                plt.close()

                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(batch_image[cnt, :], cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1
                fig.savefig(os.path.join(path + 'raw_images/', '%d_%d.png' % (epoch, step)))
                plt.close()

    saver.save(sess, '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/GAN-UNet/tmp/model.ckpt')

duration_time_sum = time.time() - start_time_sum

print("The total training time: ",elapsed(duration_time_sum))


'''
#show the loss of generaor and discriminator at every batch
'''
fig = plt.figure(figsize=(8,6))
plt.plot(steps, gen_loss_list, label='gen_loss')
plt.plot(steps, dis_loss_list, label='dis_loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('The loss of train')
plt.legend()
plt.legend(loc = 'upper right')
plt.savefig(os.path.join(path, 'loss_curve.png'))
plt.show()




















































































































































