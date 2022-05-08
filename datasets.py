"""
This file contains data sets for continual learning.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import abc

# Disable progress bar
tfds.disable_progress_bar()

# Set data directory
load_dir = "/data/public/tensorflow_datasets"


class DataSet(abc.ABC):
    # Base class for data set classes
    @abc.abstractmethod
    def __init__(self):
        pass

    def filter_fn(self, lbl, classes):
        return tf.reduce_any(tf.math.equal(lbl, classes))

    def get_split(self, classes):
        train_data = self.train_data.filter(lambda img, lbl: self.filter_fn(lbl, classes))
        if self.val_data is not None:
            val_data = self.val_data.filter(lambda img, lbl: self.filter_fn(lbl, classes))
        else:
            val_data = None
        test_data = self.test_data.filter(lambda img, lbl: self.filter_fn(lbl, classes))
        return train_data, val_data, test_data

    def get_all(self):
        return self.train_data, self.val_data, self.test_data


class SplitMNIST(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="mnist", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="mnist", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="mnist", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="mnist", data_dir=load_dir, as_supervised=True, split="test")


class SplitFashionMNIST(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="fashion_mnist", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="fashion_mnist", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="fashion_mnist", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="fashion_mnist", data_dir=load_dir, as_supervised=True, split="test")


class SplitCIFAR10(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="cifar10", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="cifar10", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="cifar10", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="cifar10", data_dir=load_dir, as_supervised=True, split="test")


class SplitCIFAR100(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="cifar100", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="cifar100", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="cifar100", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="cifar100", data_dir=load_dir, as_supervised=True, split="test")


class SplitSVHN(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="svhn_cropped", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="svhn_cropped", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="svhn_cropped", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="svhn_cropped", data_dir=load_dir, as_supervised=True, split="test")


class SplitTinyIMGNet(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="imagenet_resized/32x32", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)))
            self.val_data = tfds.load(name="imagenet_resized/32x32", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)))
        else:
            self.train_data = tfds.load(name="imagenet_resized/32x32", data_dir=load_dir, as_supervised=True, split="train")
            self.val_data = None
        self.test_data = tfds.load(name="imagenet_resized/32x32", data_dir=load_dir, as_supervised=True, split="validation")


class SplitSubImageNet0(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)),
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)),
                                      decoders={"image": tfds.decode.SkipDecoding()})
        else:
            self.train_data = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True, split="train",
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = None
        self.test_data = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True, split="validation",
                                   decoders={"image": tfds.decode.SkipDecoding()})

    def get_info(self, split):
        _, info = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True,
                                    split=split, with_info=True, decoders={"image": tfds.decode.SkipDecoding()})
        return info


class SplitSubImageNet1(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="sub_imagenet1_2012", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)),
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = tfds.load(name="sub_imagenet1_2012", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)),
                                      decoders={"image": tfds.decode.SkipDecoding()})
        else:
            self.train_data = tfds.load(name="sub_imagenet1_2012", data_dir=load_dir, as_supervised=True, split="train",
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = None
        self.test_data = tfds.load(name="sub_imagenet1_2012", data_dir=load_dir, as_supervised=True, split="validation",
                                   decoders={"image": tfds.decode.SkipDecoding()})

    def get_info(self, split):
        _, info = tfds.load(name="sub_imagenet0_2012", data_dir=load_dir, as_supervised=True,
                                    split=split, with_info=True, decoders={"image": tfds.decode.SkipDecoding()})
        return info


class SplitImageNet(DataSet):
    def __init__(self, num_validation):
        if num_validation > 0:
            self.train_data = tfds.load(name="imagenet2012", data_dir=load_dir, as_supervised=True,
                                        split="train[{:d}:]".format(int(num_validation)),
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = tfds.load(name="imagenet2012", data_dir=load_dir, as_supervised=True,
                                      split="train[:{:d}]".format(int(num_validation)),
                                      decoders={"image": tfds.decode.SkipDecoding()})
        else:
            self.train_data = tfds.load(name="imagenet2012", data_dir=load_dir, as_supervised=True, split="train",
                                        decoders={"image": tfds.decode.SkipDecoding()})
            self.val_data = None
        self.test_data = tfds.load(name="imagenet2012", data_dir=load_dir, as_supervised=True, split="validation",
                                   decoders={"image": tfds.decode.SkipDecoding()})

    def get_info(self, split):
        _, info = tfds.load(name="imagenet2012", data_dir=load_dir, as_supervised=True,
                                    split=split, with_info=True, decoders={"image": tfds.decode.SkipDecoding()})
        return info
