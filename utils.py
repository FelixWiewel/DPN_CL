"""
This file contains useful stuff
"""

import tensorflow as tf
import numpy as np


# Define training step
def get_train_step():
    dispatcher = {"softmax": softmax, "balanced_sigm": balanced_sigm}

    @tf.function
    def train_step(x_ts, y_ts, mdl_ts, opt_ts, classes_seen, classes_total, class_weights, activation,
                   prev_mdl_ts=None, prev_classes_seen=[], decay=1.0e-3, lmbd_dist=1.0):
        with tf.GradientTape() as tape:
            # Get prediciton on training data
            y_pred = mdl_ts(x_ts, training=True)
            # Get regularization
            reg = tf.add_n([tf.reduce_sum(tf.square(v)) for v in mdl_ts.trainable_variables])
            # Compute classification loss
            y_ts_sel = tf.gather(tf.one_hot(y_ts, int(classes_total)), classes_seen, axis=-1)
            y_probs = dispatcher[activation](y_pred, class_weights, classes_seen)
            class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_ts_sel, y_probs, from_logits=False))
            if prev_mdl_ts is not None:  # Distillation loss
                prev_y_pred = tf.stop_gradient(prev_mdl_ts(x_ts, training=True))
                y_dist = dispatcher[activation](prev_y_pred/2.0, class_weights, prev_classes_seen)
                y_dist_pred = dispatcher[activation](y_pred/2.0, class_weights, prev_classes_seen)
                dist_loss = 4.0*tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_dist, y_dist_pred,
                                                                                        from_logits=False))
                if activation == "balanced_sigm":  # Distill absolute precision as well
                    y_dist = dispatcher[activation](prev_y_pred, class_weights, prev_classes_seen, norm=False)
                    y_dist_pred = dispatcher[activation](y_pred, class_weights, prev_classes_seen, norm=False)
                    dist_loss += tf.reduce_mean(tf.keras.losses.mse(y_dist, y_dist_pred))
                w = float(len(prev_classes_seen))/float(len(classes_seen))
                loss = (1.0 - w)*class_loss + w*lmbd_dist*dist_loss + decay*reg
            else:
                loss = class_loss + decay*reg
        grads = tape.gradient(loss, mdl_ts.trainable_variables)
        opt_ts.apply_gradients(zip(grads, mdl_ts.trainable_variables))
        return loss
    return train_step


def softmax(pred, class_weights, classes_seen, norm=True):
    if norm:
        return tf.nn.softmax(tf.gather(pred, classes_seen, axis=-1), axis=-1)
    else:
        return tf.math.exp(tf.gather(pred, classes_seen, axis=-1))


def balanced_sigm(pred, class_weights, classes_seen, norm=True):
    logits = tf.gather(tf.nn.sigmoid(pred)*tf.expand_dims(class_weights, axis=0), classes_seen, axis=-1)
    if norm:
        return logits/tf.reduce_sum(logits, axis=-1, keepdims=True)
    else:
        return logits


# Define evaluation function
def evaluate(eval_ds, mdl_eval, classes_seen, classes_total):
    m_acc = tf.keras.metrics.CategoricalAccuracy()
    m_loss = tf.keras.metrics.Mean()
    for x_eval, y_eval in eval_ds:
        y_eval_sel = tf.gather(tf.one_hot(y_eval, int(classes_total)), classes_seen, axis=-1)
        y_pred = tf.gather(mdl_eval(x_eval, training=False), classes_seen, axis=-1)
        m_loss.update_state(tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_eval_sel, y_pred, from_logits=True)))
        m_acc.update_state(y_eval_sel, tf.nn.softmax(y_pred, axis=-1))
    return m_loss.result(), m_acc.result()


# Get precisions
def get_precisions(eval_ds, ood_ds, mdl_eval, classes_seen, activation):
    dispatcher = {"softmax": softmax, "balanced_sigm": balanced_sigm}
    precisions = []
    max_prob = []
    labels = []
    for x_eval, y_eval in eval_ds:
        y_pred = dispatcher[activation](mdl_eval(x_eval, training=False), 1.0, classes_seen, norm=False)
        precisions.append(tf.reduce_sum(y_pred, axis=-1))
        y_pred = dispatcher[activation](mdl_eval(x_eval, training=False), 1.0, classes_seen, norm=True)
        max_prob.append(tf.reduce_max(y_pred, axis=-1))
        labels.append(tf.ones((x_eval.shape[0],), dtype=tf.float32))
    for x_ood_eval, _ in ood_ds:
        y_pred = dispatcher[activation](mdl_eval(x_ood_eval, training=False), 1.0, classes_seen, norm=False)
        precisions.append(tf.reduce_sum(y_pred, axis=-1))
        y_pred = dispatcher[activation](mdl_eval(x_ood_eval, training=False), 1.0, classes_seen, norm=True)
        max_prob.append(tf.reduce_max(y_pred, axis=-1))
        labels.append(tf.zeros((x_ood_eval.shape[0],), dtype=tf.float32))
    return tf.concat(precisions, axis=0), tf.concat(max_prob, axis=0), tf.concat(labels, axis=0)


# Shuffle class order
def shuffle_cls_order(tasks, seed):
    classes = np.unique(np.concatenate(tasks, axis=0))
    np.random.seed(seed)
    np.random.shuffle(classes)
    shuffle_tasks = [classes[idx].tolist() for idx in tasks]
    return shuffle_tasks


# Update the class weights
def update_cls_weights(class_weights, buf_class_weights, new_bsz, rehearsal_bsz, batch_size):
    new_class_weights = (class_weights*new_bsz + buf_class_weights*rehearsal_bsz)/batch_size
    return new_class_weights


def get_ds_info(ds, classes_total):
    class_weights = tf.zeros(classes_total)
    ds_size = 0

    for x, y in ds:
        ds_size += 1
        class_weights += tf.cast(tf.one_hot(y, classes_total), dtype=tf.float32)
    return ds_size, class_weights
