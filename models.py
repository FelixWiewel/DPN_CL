"""
This file contains models.
"""

import tensorflow as tf


class ResBlock(tf.keras.layers.Layer):
    """
    Original residual block
    """

    def __init__(self, filters, kernel_size, stride=1, projection=False, last=False):
        super(ResBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.projection = projection
        self.last = last
        self.conv0 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.stride, padding="SAME", use_bias=True)
        self.b_norm0 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding="SAME", use_bias=True)
        if self.projection:
            self.conv2 = tf.keras.layers.Conv2D(self.filters, 1, self.stride, padding="SAME", use_bias=True)
            self.b_norm2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.b_norm1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs, training=None, mask=None):
        skip = inputs
        output = self.conv0(inputs)
        output = self.b_norm0(output, training)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.b_norm1(output, training)
        if self.projection:
            skip = self.conv2(skip)
            skip = self.b_norm2(skip, training)
        else:
            if self.stride == 2:
                skip = skip[:, ::2, ::2, :]
                skip = tf.concat((skip, tf.zeros_like(skip, dtype=tf.float32)), axis=-1)
        output = tf.add(output, skip)
        if not self.last:
            output = self.relu(output)
        return output


class ResNet18(tf.keras.Model):
    """
    Standard ResNet18
    """

    def __init__(self, n):
        super(ResNet18, self).__init__()
        self.n = n
        self.conv0 = tf.keras.layers.Conv2D(64, 7, activation="linear", strides=2, padding="SAME", use_bias=True)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")
        self.b_norm = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu = tf.nn.relu
        self.res0 = ResBlock(64, 3, 1)
        self.res1 = ResBlock(64, 3, 1)
        self.res2 = ResBlock(128, 3, 2, projection=True)
        self.res3 = ResBlock(128, 3, 1)
        self.res4 = ResBlock(256, 3, 2, projection=True)
        self.res5 = ResBlock(256, 3, 1)
        self.res6 = ResBlock(512, 3, 2, projection=True)
        self.res7 = ResBlock(512, 3, 1, last=True)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.n, activation="linear")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        output = self.conv0(inputs)
        output = self.b_norm(output, training)
        output = self.relu(output)
        output = self.pool(output)
        output = self.res0(output, training)
        output = self.res1(output, training)
        output = self.res2(output, training)
        output = self.res3(output, training)
        output = self.res4(output, training)
        output = self.res5(output, training)
        output = self.res6(output, training)
        output = self.res7(output, training)
        output = self.avg_pool(output)
        output = self.dense(output)
        return output


class ResNet32(tf.keras.Model):
    """
    Simple ResNet32 for CIFAR10
    """

    def __init__(self, n):
        super(ResNet32, self).__init__()
        self.n = n
        self.conv0 = tf.keras.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=True)
        self.b_norm = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu = tf.nn.relu
        self.res0 = ResBlock(16, 3, 1)
        self.res1 = ResBlock(16, 3, 1)
        self.res2 = ResBlock(16, 3, 1)
        self.res3 = ResBlock(16, 3, 1)
        self.res4 = ResBlock(16, 3, 1)
        self.res5 = ResBlock(32, 3, 2)
        self.res6 = ResBlock(32, 3, 1)
        self.res7 = ResBlock(32, 3, 1)
        self.res8 = ResBlock(32, 3, 1)
        self.res9 = ResBlock(32, 3, 1)
        self.res10 = ResBlock(64, 3, 2)
        self.res11 = ResBlock(64, 3, 1)
        self.res12 = ResBlock(64, 3, 1)
        self.res13 = ResBlock(64, 3, 1)
        self.res14 = ResBlock(64, 3, 1, last=True)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.n, activation="linear")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        output = self.conv0(inputs)
        output = self.b_norm(output, training)
        output = self.relu(output)
        output = self.res0(output, training)
        output = self.res1(output, training)
        output = self.res2(output, training)
        output = self.res3(output, training)
        output = self.res4(output, training)
        output = self.res5(output, training)
        output = self.res6(output, training)
        output = self.res7(output, training)
        output = self.res8(output, training)
        output = self.res9(output, training)
        output = self.res10(output, training)
        output = self.res11(output, training)
        output = self.res12(output, training)
        output = self.res13(output, training)
        output = self.res14(output, training)
        output = self.avg_pool(output)
        output = self.dense(output)
        return output
