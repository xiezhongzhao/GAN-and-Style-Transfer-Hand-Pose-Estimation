import tensorflow as tf


# Input(-1, 128, 128)->Output(-1, 0/1)
def discriminator(unet_image, reuse=None):

    with tf.variable_scope('discriminator', reuse=reuse):

        n = 32

        unet_image = tf.reshape(unet_image,[-1,128,128,1])

        # original image(128,128,1)->(64,64,32)
        dis_conv1 = tf.layers.batch_normalization(
            tf.layers.conv2d(unet_image, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        # image size(64,64,32)->(32,32,32)
        dis_conv2 = tf.layers.batch_normalization(
            tf.layers.conv2d(dis_conv1, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        # # to be later used for recognition or other discriminative purpose
        dis_hidden_layer = dis_conv2
        #
        # image size(32,32,32)->(16,16,32)
        dis_conv3 = tf.layers.batch_normalization(
            tf.layers.conv2d(dis_conv2, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        # image size(16,16,32)->(8,8,32)
        dis_conv4 = tf.layers.batch_normalization(
            tf.layers.conv2d(dis_conv3, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        # image size(8,8,32)->(4,4,32)
        dis_conv5 = tf.layers.batch_normalization(
            tf.layers.conv2d(dis_conv4, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))
        flat = tf.layers.flatten(dis_conv5)

        x = tf.layers.dense(flat, 128)

        d_out_logits = tf.layers.dense(x, 1)

        d_out = tf.nn.sigmoid(d_out_logits)

        return d_out, d_out_logits, dis_hidden_layer


#from the shape of hiddde layer is [-1,32,32,32] to the joints of size [-1, 42]
def posterior_recognition(dis_hidden_layer=None, reuse=None):

    with tf.variable_scope('posterior', reuse=reuse):

        n =  32

        # image size(32,32,32)->(16,16,32)
        dis_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(dis_hidden_layer, filters=n, kernel_size=6, strides=2, padding="same",
                                                  activation=tf.nn.leaky_relu))

        # image size(16,16,32)->(8,8,32)
        dis_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv3, filters=n, kernel_size=6, strides=2, padding="same",
                                                  activation=tf.nn.leaky_relu))

        # image size(8,8,32)->(4,4,32)
        dis_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv4, filters=n, kernel_size=6, strides=2, padding="same",
                                                  activation=tf.nn.leaky_relu))

        flat = tf.layers.flatten(dis_conv5)

        x = tf.layers.dense(flat, 32 * 4 * 4)

        d_out_joint = tf.layers.dense(x, 42)

        return d_out_joint




