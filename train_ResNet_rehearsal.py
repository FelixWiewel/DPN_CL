"""
Simple script for training ResNet with rehearsal
"""

import tensorflow as tf
import models
import buffers
import utils
import datasets
import augmentations
import gin
import argparse
import copy
from ray import tune


@gin.configurable("run", denylist=["base_dir"])
def run(epochs, val_iters, batch_size, val_batches, learning_rates, boundaries, seed, tasks, arch, buffer_sizes,
        buffer_mode, buffer_growth, oversampling, dataset, runs, distillation, dist_strength, weight_decay, activation,
        ood_dataset, base_dir, restart_sched=True, tune_hp=False):
    # Run multiple runs
    for r in range(runs):
        # Shuffle class order
        shuffle_tasks = utils.shuffle_cls_order(tasks, seed)

        print("Starting run {}".format(r))
        # Train with multiple buffer sizes
        for b in buffer_sizes:
            # Instantiate model, optimizer and buffer
            print("Instantiate {} for training on {} with a buffer size of {}, growth set to {} and {} sample"
                  " selection..." .format(arch, dataset, b, buffer_growth, buffer_mode))
            if arch == "ResNet32":
                if dataset == "CIFAR10":
                    mdl = models.ResNet32(10)
                elif dataset == "CIFAR100":
                    mdl = models.ResNet32(100)
                elif dataset == "ImageNet":
                    mdl = models.ResNet32(1000)
            if arch == "ResNet18":
                if dataset == "CIFAR10":
                    mdl = models.ResNet18(10)
                elif dataset == "CIFAR100" or dataset == "SubImageNet0" or dataset == "SubImageNet1":
                    mdl = models.ResNet18(100)
                elif dataset == "ImageNet":
                    mdl = models.ResNet18(1000)
            buf = buffers.ClassBalanceBuffer(b, buffer_mode)

            # Create tensorboard writer
            if not tune_hp:
                log_dir = base_dir + "/run_{}_buffer_{}".format(r, b)
                fw = tf.summary.create_file_writer(log_dir)

            # Train multiple tasks
            classes_seen = []
            classes_total = sum([len(t) for t in shuffle_tasks])
            for i, t in enumerate(shuffle_tasks):
                # Increase counter of seen classes
                prev_classes_seen = classes_seen
                classes_seen = classes_seen + t
                if buffer_growth:  # Increase buffer size
                    buf.max_buffer_size = len(classes_seen)*b
                # Load data
                print("Loading data for task {}...".format(t))
                if dataset == "CIFAR10":
                    train, val, _ = datasets.SplitCIFAR10(num_validation=val_batches*batch_size).get_split(t)
                elif dataset == "CIFAR100":
                    train, val, _ = datasets.SplitCIFAR100(num_validation=val_batches*batch_size).get_split(t)
                elif dataset == "SubImageNet0":
                    train, val, _ = datasets.SplitSubImageNet0(num_validation=val_batches*batch_size).get_split(t)
                elif dataset == "SubImageNet1":
                    train, val, _ = datasets.SplitSubImageNet1(num_validation=val_batches*batch_size).get_split(t)
                elif dataset == "ImageNet":
                    train, val, _ = datasets.SplitImageNet(num_validation=val_batches*batch_size).get_split(t)
                # Set preprocessing and augmentation functions
                if (dataset == "CIFAR100") or (dataset == "CIFAR10"):
                    pre_proc_func_train = augmentations.pre_proc_cifar
                    pre_proc_func_val = augmentations.pre_proc_cifar
                    pre_proc_func_buf = augmentations.pre_proc_cifar
                    augm_func = augmentations.resnet_cifar10
                elif dataset == "ImageNet" or dataset == "SubImageNet0" or dataset == "SubImageNet1":
                    info_train = datasets.SplitImageNet(num_validation=val_batches*batch_size).get_info("train")
                    info_val = datasets.SplitImageNet(num_validation=val_batches*batch_size).get_info("validation")
                    pre_proc_func_train = lambda img, lbl: augmentations.pre_proc_imagenet_train(img, lbl, info_train)
                    pre_proc_func_val = lambda img, lbl: augmentations.pre_proc_imagenet_val(img, lbl, info_val)
                    pre_proc_func_buf = lambda img, lbl: augmentations.pre_proc_imagenet_train(img, lbl, info_train)
                    augm_func = augmentations.resnet_imagenet

                # Get information on the training dataset
                print("Getting information on training dataset...")
                train_ds_size, class_weights = utils.get_ds_info(train, classes_total)

                # Compute batch sizes for oversampling
                if t != shuffle_tasks[0]:
                    if oversampling is None:
                        new_bsz = int(batch_size*train_ds_size/(train_ds_size+b))
                        rehearsal_bsz = int(tf.math.maximum(batch_size - new_bsz, 1))
                    elif oversampling == "buffer":
                        new_bsz = int(batch_size/2)
                        rehearsal_bsz = int(batch_size - new_bsz)
                    elif oversampling == "class":
                        cls_buf, _ = tf.unique(buf.y_buffer)
                        cls_buf = tf.cast(cls_buf.shape[0], dtype=tf.float32)
                        cls_train_ds = tf.cast(tf.math.count_nonzero(class_weights), dtype=tf.float32)
                        new_bsz = int(batch_size*cls_train_ds/cls_buf)
                        rehearsal_bsz = int(tf.math.maximum(batch_size - new_bsz, 1))
                else:
                    new_bsz = batch_size
                    rehearsal_bsz = None
                # Prepare training and validation datasets
                train_ds = train.repeat().shuffle(5000).map(pre_proc_func_train, num_parallel_calls=tf.data.AUTOTUNE)\
                    .map(augm_func, num_parallel_calls=tf.data.AUTOTUNE)
                if t != shuffle_tasks[0]:
                    buf_ds = buf.create_tf_ds().repeat().shuffle(b)\
                        .map(pre_proc_func_buf, num_parallel_calls=tf.data.AUTOTUNE).map(augm_func, num_parallel_calls=tf.data.AUTOTUNE)
                if (dataset == "CIFAR100") or (dataset == "CIFAR10"):
                    train_ds = train_ds.batch(new_bsz).prefetch(100)
                    if t != shuffle_tasks[0]:
                        buf_ds = buf_ds.batch(rehearsal_bsz).prefetch(100)
                elif dataset == "ImageNet" or dataset == "SubImageNet0" or dataset == "SubImageNet1":
                    train_ds = train_ds.batch(new_bsz).prefetch(100)
                    if t != shuffle_tasks[0]:
                        buf_ds = buf_ds.batch(rehearsal_bsz).prefetch(100)
                train_ds_iter = train_ds.as_numpy_iterator()
                if val is not None:
                    val_ds = val.map(pre_proc_func_val, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(100)
                else:
                    val_ds = None
                # Prepare rehearsal buffer
                if t != shuffle_tasks[0]:
                    buf_ds_iter = buf_ds.as_numpy_iterator()
                else:
                    buf_ds_iter = None

                if t != shuffle_tasks[0]:
                    # Update class weights
                    print("Updating class weights")
                    buf_ds_size, buf_class_weights = utils.get_ds_info(buf.create_tf_ds(), classes_total)
                    class_weights = utils.update_cls_weights(class_weights, buf_class_weights, new_bsz, rehearsal_bsz, batch_size)

                # Compute number of iterations based on epochs
                iters = round(epochs*train_ds_size/new_bsz)

                # Instantiate optimizer
                sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([round(iters * b) for b in boundaries],
                                                                             learning_rates)
                if t != shuffle_tasks[0] and restart_sched is False:
                    sched = learning_rates[-1]
                opt = tf.keras.optimizers.SGD(sched, momentum=0.9)

                # Get training step
                t_step = utils.get_train_step()

                # Train model
                print("Training for {} iterations with new/rehearsal batch size {}/{}...".format(iters, new_bsz, rehearsal_bsz))
                for step in range(iters):
                    x_t, y_t = next(train_ds_iter)
                    if buf_ds_iter is not None:
                        x_r, y_r = next(buf_ds_iter)
                        x_t = tf.concat((x_t, x_r), axis=0)
                        y_t = tf.concat((y_t, y_r), axis=0)
                    if distillation and (t != shuffle_tasks[0]):
                        # Perform training step with distillation
                        l = t_step(x_t, y_t, mdl, opt, classes_seen, classes_total, class_weights, activation,
                                   prev_mdl, prev_classes_seen, decay=weight_decay, lmbd_dist=dist_strength)
                    else:
                        # Perform training step
                        l = t_step(x_t, y_t, mdl, opt, classes_seen, classes_total, class_weights, activation,
                                   decay=weight_decay, lmbd_dist=dist_strength)
                    if step % val_iters == 0:
                        print("Step: {} Train Loss: {:.3}".format(step, l))
                        if val_batches != 0:
                            val_loss, val_acc = utils.evaluate(val_ds, mdl, classes_seen, classes_total)
                            if tune_hp:
                                tune.report(val_acc=val_acc.numpy(), iteration=step)
                            else:
                                print("Val. Loss: {:.3} Val. Acc.: {:.3}".format(val_loss, val_acc))
                                with fw.as_default():
                                    name = base_dir.split("/")[-1]
                                    tf.summary.scalar("train_loss_{}".format(name), l, step)
                                    tf.summary.scalar("learning_rate_{}".format(name), sched(step), step)
                                    tf.summary.scalar("validation_loss_{}".format(name), val_loss, step=step)
                                    tf.summary.scalar("validation_acc_{}".format(name), val_acc, step=step)
                # Update buffer with new classes
                if not tune_hp:
                    print("Filling/updating buffer...")
                    train_ds = train.batch(batch_size).prefetch(10)
                    for x_u, y_u in train_ds:
                        # Fill buffer
                        if (activation == "balanced_sigm") and (buffer_mode != "random"):
                            x_u_pp, _ = pre_proc_func_buf(x_u, y_u)
                            pred_u = mdl(x_u_pp, training=False)
                            p_u = tf.reduce_sum(utils.balanced_sigm(pred_u, 1.0, classes_seen, norm=False), axis=-1)
                        else:
                            p_u = tf.ones((x_u.shape[0],))
                        if not buf.is_full():
                            buf.add_sample(x_u, y_u, p_u)
                        else:
                            buf.update_buffer(x_u, y_u, p_u)
                    buf.summary()
                    # Save model as checkpoint
                    print("Saving model checkpoint...")
                    mdl.save_weights(log_dir + "/task_{}".format(i))
                    if distillation:
                        # Copy model for distillation in next step
                        prev_mdl = copy.deepcopy(mdl)


if __name__ == "__main__":
    # Parse config
    parser = argparse.ArgumentParser(description="Train ResNet using rehearsal")
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config+"/config.gin")
    run(base_dir=args.config)
