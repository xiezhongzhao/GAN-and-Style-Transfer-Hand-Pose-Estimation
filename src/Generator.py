import tensorflow as tf

# Input(-1, 42)->Output(-1, 128, 128, 1)
def generator(joints_in, reuse=None):  # , noise_in

    with tf.variable_scope('generator', reuse=reuse):

        n = 32

        joints = tf.reshape(joints_in, [-1, 42])

        input_map = tf.layers.batch_normalization(
            tf.layers.dense(joints, units=4 * 4 * 32, activation=tf.nn.leaky_relu))

        x_in = tf.reshape(input_map, shape=[-1, 4, 4, 32])

        # noise(-1, 42)->(-1,4 * 4 * 32)
        x = tf.layers.batch_normalization(tf.layers.dense(x_in, units=4 * 4 * 32, activation=tf.nn.leaky_relu))

        # image(-1,4,4,32)->(-1,8,8,32)
        gen1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(x,
                                                                        filters=n, kernel_size=6, strides=2,
                                                                        use_bias=True,
                                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                                            stddev=0.01),
                                                                        padding='same', activation=tf.nn.leaky_relu))

        # image(-1,8,8,32)->(-1,16,16,32)
        gen2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen1,
                                                                        filters=n, kernel_size=6, strides=2,
                                                                        use_bias=True,
                                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                                            stddev=0.01),
                                                                        padding='same', activation=tf.nn.leaky_relu))

        # image(-1,16,16,32)->(-1,32,32,32)
        gen3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen2,
                                                                        filters=n, kernel_size=6, strides=2,
                                                                        use_bias=True,
                                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                                            stddev=0.01),
                                                                        padding='same', activation=tf.nn.leaky_relu))

        # image(-1,32,32,32)->(1, 64, 64, 32)
        gen4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen3,
                                                                        filters=n, kernel_size=6, strides=2,
                                                                        use_bias=True,
                                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                                            stddev=0.01),
                                                                        padding='same', activation=tf.nn.leaky_relu))

        # image(1, 64, 64, 32)->(1, 128, 128, 1)
        gen5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen4,
                                                                        filters=1, kernel_size=6, strides=2,
                                                                        use_bias=True,
                                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                                            stddev=0.01),
                                                                        padding='same', activation=tf.nn.tanh))

        return gen5











