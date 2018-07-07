import tensorflow as tf

def unet(gen_image, reuse=None):

    with tf.variable_scope('unet', reuse=reuse):

        n = 32

        # original image(128,128,1)->(64,64,32)
        dis_conv1 = tf.layers.batch_normalization(
            tf.layers.conv2d(gen_image, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        # image size(64,64,32)->(32,32,32)
        dis_conv2 = tf.layers.batch_normalization(
            tf.layers.conv2d(dis_conv1, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

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

        # image size (4,4,32)->(8,8,32)
        gen1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(dis_conv5, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))
        # image size (8,8,32)->(8,8,64)
        gen1_con = tf.concat([dis_conv4, gen1], 3)

        # image size (8,8,64)->(16,16,32)
        gen2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen1_con, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))
        # image size (16,16,32)->(16,16,64)
        gen2_con = tf.concat([dis_conv3, gen2], 3)

        # image size (16,16,64)->(32,32,32)
        gen3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen2_con, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))
        # image size (32,32,32)->(32,32,64)
        gen3_con = tf.concat([dis_conv2, gen3], 3)

        # image size (32,32,32)->(64,64,32)
        gen4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen3_con, filters=n, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))
        # image size (64,64,32)->(64,64,64)
        gen4_con = tf.concat([dis_conv1, gen4], 3)

        # image size (64,64,64)->(128,128,1)
        gen4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen4_con, filters=1, kernel_size=6, strides=2, padding="same",
                             activation=tf.nn.leaky_relu))

        img_unet = gen4

        return img_unet




















