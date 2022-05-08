"""
Simple script for training ResNet with rehearsal
"""

import models
import datasets
import utils
import gin
import argparse
import numpy as np
import augmentations
import tensorflow as tf
import matplotlib.pyplot as plt

# Set style and size
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14,
    "figure.figsize": (8, 4)})


@gin.configurable("run", denylist=["base_dir"])
def run(epochs, val_iters, batch_size, val_batches, learning_rates, boundaries, seed, tasks, arch, buffer_sizes,
        buffer_mode, buffer_growth, oversampling, dataset, runs, distillation, dist_strength, weight_decay, activation,
        ood_dataset, base_dir, restart_sched=True, tune_hp=False):
    # Create metric matrices
    r_loss = np.zeros((runs, len(buffer_sizes), len(tasks)), dtype=float)
    r_acc = np.zeros((runs, len(buffer_sizes), len(tasks)), dtype=float)
    if ood_dataset is not None:
        r_auc = np.zeros((runs, len(buffer_sizes), len(tasks)), dtype=float)
        r_auc_base = np.zeros((runs, len(buffer_sizes), len(tasks)), dtype=float)

    # Run multiple runs
    for r in range(runs):
        # Shuffle class order
        shuffle_tasks = utils.shuffle_cls_order(tasks, seed)

        print("Starting run {}".format(r))
        # Train with multiple buffer sizes
        for h, b in enumerate(buffer_sizes):
            # Instantiate model, optimizer and buffer
            print("Instantiate {} for testing on {} with a buffer size of {}, growth set to {} and {} sample"
                  " selection...".format(arch, dataset, b, buffer_growth, buffer_mode))
            if arch == "ResNet32":
                if dataset == "CIFAR10":
                    mdl = models.ResNet32(10)
                elif dataset == "CIFAR100":
                    mdl = models.ResNet32(100)
                elif dataset == "ImageNet":
                    mdl = models.ResNet32(1000)
                mdl.build((None, 32, 32, 3))
            if arch == "ResNet18":
                if dataset == "CIFAR10":
                    mdl = models.ResNet18(10)
                    mdl.build((None, 32, 32, 3))
                elif dataset == "CIFAR100" or dataset == "SubImageNet0" or dataset == "SubImageNet1":
                    mdl = models.ResNet18(100)
                    mdl.build((None, 224, 224, 3))
                elif dataset == "ImageNet":
                    mdl = models.ResNet18(1000)
                    mdl.build((None, 224, 224, 3))

            # Create log dir
            log_dir = base_dir + "/run_{}_buffer_{}".format(r, b)

            # Test multiple tasks
            classes_seen = []
            classes_total = sum([len(t) for t in shuffle_tasks])
            for i, t in enumerate(shuffle_tasks):
                # Increase counter of seen classes
                classes_seen = classes_seen + t
                # Load model
                load_dir = log_dir + "/task_{}".format(i)
                print("Loading checkpoint from {}".format(load_dir))
                mdl.load_weights(load_dir)

                # Load data
                load_tasks = [item for sublist in shuffle_tasks[0:i+1] for item in sublist]
                print("Loading data for task {}".format(load_tasks))
                if dataset == "CIFAR10":
                    _, _, test = datasets.SplitCIFAR10(num_validation=val_batches * batch_size).get_split(load_tasks)
                    test_ds = test.map(augmentations.pre_proc_cifar).batch(batch_size).prefetch(10)
                elif dataset == "CIFAR100":
                    _, _, test = datasets.SplitCIFAR100(num_validation=val_batches * batch_size).get_split(load_tasks)
                    test_ds = test.map(augmentations.pre_proc_cifar).batch(batch_size).prefetch(10)
                elif dataset == "SubImageNet0":
                    _, _, test = datasets.SplitSubImageNet0(num_validation=val_batches*batch_size).get_split(load_tasks)
                    info_val = datasets.SplitSubImageNet0(num_validation=val_batches*batch_size).get_info("validation")
                    pre_proc_func_val = lambda img, lbl: augmentations.pre_proc_imagenet_val(img, lbl, info_val)
                    test_ds = test.map(pre_proc_func_val).batch(batch_size).prefetch(10)
                elif dataset == "SubImageNet1":
                    _, _, test = datasets.SplitSubImageNet1(num_validation=val_batches*batch_size).get_split(load_tasks)
                    info_val = datasets.SplitSubImageNet1(num_validation=val_batches*batch_size).get_info("validation")
                    pre_proc_func_val = lambda img, lbl: augmentations.pre_proc_imagenet_val(img, lbl, info_val)
                    test_ds = test.map(pre_proc_func_val).batch(batch_size).prefetch(10)
                elif dataset == "ImageNet":
                    _, _, test = datasets.SplitImageNet(num_validation=val_batches*batch_size).get_split(load_tasks)
                    info_val = datasets.SplitImageNet(num_validation=val_batches*batch_size).get_info("validation")
                    pre_proc_func_val = lambda img, lbl: augmentations.pre_proc_imagenet_val(img, lbl, info_val)
                    test_ds = test.map(pre_proc_func_val).batch(batch_size).prefetch(10)

                # Load ood dataset if specified
                if ood_dataset is None:
                    ood_test_ds = None
                else:
                    if ood_dataset == "CIFAR10":
                        _, _, ood_test = datasets.SplitCIFAR10(num_validation=val_batches * batch_size).get_all()
                    elif ood_dataset == "SVHN":
                        _, _, ood_test = datasets.SplitSVHN(num_validation=val_batches * batch_size).get_all()
                    elif ood_dataset == "TinyIMGNet":
                        _, _, ood_test = datasets.SplitTinyIMGNet(num_validation=val_batches*batch_size).get_all()
                    ood_test_ds = ood_test.map(augmentations.pre_proc_cifar, num_parallel_calls=tf.data.AUTOTUNE)\
                        .shuffle(5000).batch(batch_size).prefetch(100)

                # Test models
                if ood_dataset is not None:
                    prec, max_prob, lbls = utils.get_precisions(test_ds, ood_test_ds, mdl, classes_seen, activation)
                    if i == 0:
                        plt.hist(prec[lbls == 1].numpy(), bins=100, alpha=0.75, log=True)
                        plt.hist(prec[lbls == 0].numpy(), bins=100, alpha=0.75, log=True)
                        plt.legend(["ID", "OOD"])
                        plt.xlabel(r"Precision $\alpha_{0}$")
                        plt.ylabel(r"Occurrence")
                        plt.tight_layout()
                        plt.savefig(fname=base_dir+"/precision_{}_{}_{}_{}.pdf".format(ood_dataset, r, h, i), format="pdf")
                        plt.show()
                    m_auc = tf.keras.metrics.AUC()
                    m_auc_base = tf.keras.metrics.AUC()
                    m_auc.update_state(lbls, tf.clip_by_value(prec, 0.0, tf.reduce_max(prec))/tf.reduce_max(prec))
                    m_auc_base.update_state(lbls, max_prob)
                    auc = m_auc.result()
                    auc_base = m_auc_base.result()
                    print("AUC: {:.3}".format(auc))
                    print("AUC Baseline: {:.3}".format(auc_base))
                    r_auc[r, h, i] = auc
                    r_auc_base[r, h, i] = auc_base
                loss, acc = utils.evaluate(test_ds, mdl, classes_seen, classes_total)
                print("Loss: {:.3} Accuracy: {:.3}".format(loss, acc))
                r_loss[r, h, i] = loss
                r_acc[r, h, i] = acc

            # Save metrics
            np.save(base_dir+"/r_loss.npy", r_loss)
            np.save(base_dir+"/r_acc.npy", r_acc)
            if ood_dataset is not None:
                np.save(base_dir+"/r_auc.npy", r_auc)
                np.save(base_dir+"/r_auc_base.npy", r_auc_base)


if __name__ == "__main__":
    # Parse config
    parser = argparse.ArgumentParser(description="Test ResNet using rehearsal")
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config + "/config.gin")
    run(base_dir=args.config)
