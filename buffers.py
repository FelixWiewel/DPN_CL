"""
This file contains buffer implementations
"""

import abc
import numpy as np
import tensorflow as tf


class Buffer(abc.ABC):
    """
    A memory that randomly selects samples from a batch to update the buffer with.
    """

    def __init__(self, max_buffer_size=1000, mode="random"):
        self.x_buffer = []
        self.y_buffer = []
        self.max_buffer_size = max_buffer_size
        self.mode = mode
        self.ds = None
        self.ds_iter = None
        super(Buffer, self).__init__()

    def add_sample(self, x_batch, y_batch, p_batch):
        # Check is full and if not add a sample
        if not self.is_full():
            batch_size = x_batch.shape[0]
            max_add = tf.clip_by_value(self.max_buffer_size - len(self.x_buffer), 0, batch_size)
            self.x_buffer += tf.unstack(x_batch[0:max_add], axis=0)
            self.y_buffer += tf.unstack(y_batch[0:max_add], axis=0)
            self.p_buffer += tf.unstack(p_batch[0:max_add], axis=0)

    def is_full(self):
        # Check if buffer is full or not
        if len(self.x_buffer) < self.max_buffer_size:
            return False
        else:
            return True

    @abc.abstractmethod
    def update_buffer(self, x, y, p):
        pass

    def summary(self):
        print("+======================================+")
        print("| Summary                              |")
        print("+======================================+")
        print("| Number of samples in memory: {}".format(len(self.x_buffer)))
        print("+--------------------------------------+")
        cl, counts = np.unique(self.y_buffer, return_counts=True)
        for i, j in zip(cl, counts):
            print("| Class {}: {}".format(i, j))
        print("+--------------------------------------+")

    def create_tf_ds(self):
        return tf.data.Dataset.from_tensor_slices((self.x_buffer, self.y_buffer))


class ClassBalanceBuffer(Buffer):
    """
    A buffer that keeps the number of samples per class balanced but only selects examples with highest precision.
    """

    def __init__(self, max_buffer_size=1000, mode="random"):
        self.p_buffer = []
        super(ClassBalanceBuffer, self).__init__(max_buffer_size, mode)

    def update_buffer(self, x_batch, y_batch, p_batch):
        # Select the sample of the majority class with lowest precision to be replaced
        for x, y, p in zip(tf.unstack(x_batch, axis=0), tf.unstack(y_batch, axis=0), tf.unstack(p_batch, axis=0)):
            # Get indices of majority class

            classes, idx, counts = tf.unique_with_counts(self.y_buffer)
            majority_class = classes[tf.argmax(counts)]
            majority_idx = tf.range(0, len(self.x_buffer))[self.y_buffer == majority_class]
            if self.mode == "random":
                repl_idx = np.random.choice(majority_idx)
            elif self.mode == "max_precision":
                min_idx = np.argmin(np.take(self.p_buffer, majority_idx))
                repl_idx = np.take(majority_idx, min_idx)
            elif self.mode == "min_precision":
                max_idx = np.argmax(np.take(self.p_buffer, majority_idx))
                repl_idx = np.take(majority_idx, max_idx)
            elif self.mode == "balanced_precision":
                dist = np.square(np.take(self.p_buffer, majority_idx) - p)
                min_idx = np.argmin(dist)
                repl_idx = np.take(majority_idx, min_idx)
            self.x_buffer[repl_idx] = x
            self.y_buffer[repl_idx] = y
            self.p_buffer[repl_idx] = p
