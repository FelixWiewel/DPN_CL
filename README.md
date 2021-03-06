# DPN_CL

Code repository for the paper "Dirichlet Prior Networks for Continual Learning" accepted for publication at IJCNN 2022.

## Running experiments
Experiemnts are configured using gin-config and therefore can easily be created and adjusted using config files. An example for such a configuration is given by:

```
# Hyperparameters
run.epochs = 250
run.val_iters = 500
run.batch_size = 256
run.val_batches = 0
run.learning_rates = [0.1, 0.01]
run.boundaries = [0.7]
run.seed = 1993
run.tasks = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
run.arch = "ResNet32"
run.buffer_sizes = [20,]
run.buffer_mode = "random"
run.buffer_growth = True
run.oversampling = None
run.dataset = "CIFAR100"
run.runs = 5
run.distillation = True
run.dist_strength = 2.0
run.weight_decay = 2e-4
run.activation = "softmax"
run.ood_dataset = "CIFAR10"
```

This experiment can then be run by exceuting:

```
python -u train_ResNet_rehearsal.py --config="./experiments/CIFAR100_10/RN32_RD"
```
