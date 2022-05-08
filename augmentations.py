"""
Data augmentations used for contrastive learning
"""

import tensorflow as tf


# Preprocessing functions
def pre_proc_cifar(img, lbl):
    img = tf.cast(img, tf.float32)/255.0
    lbl = tf.cast(lbl, tf.int32)
    return img, lbl


def pre_proc_imagenet_val(img, lbl, ds_info):
    img = ds_info.features["image"].decode_example(img)
    img = tf.cast(tf.keras.preprocessing.image.smart_resize(img, (256, 256)), tf.float32)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="torch")
    img = tf.image.central_crop(img, 0.875)
    lbl = tf.cast(lbl, tf.int32)
    return img, lbl


def pre_proc_imagenet_train(img, lbl, ds_info):
    img = ds_info.features["image"].decode_example(img)
    img = tf.cast(tf.keras.preprocessing.image.smart_resize(img, (256, 256)), tf.float32)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="torch")
    lbl = tf.cast(lbl, tf.int32)
    return img, lbl


# Augmentations
def resnet_cifar10(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], "SYMMETRIC")
    img = tf.image.random_crop(img, (32, 32, 3))
    return img, lbl


def resnet_imagenet(img, lbl):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, (224, 224, 3))
    return img, lbl