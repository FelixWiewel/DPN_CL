"""
This script visualizes the results of the experiments
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style and size
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14,
    "figure.figsize": (8, 4)})

# Open all files and plot results: Acc. over task for 5 steps
print("+----------------------------+")
print("| CIFAR100                   |")
print("+----------------------------+")
print("Results for 5 steps:")
path = "experiments/CIFAR100_5/"
files = ["RN32_RD", "RN32_RD_BOS", "RN32_RD_COS", "RN32_RDB", "RN32_RDB_BOS", "RN32_RDB_COS"]
legends = ["No OS", "BOS", "COS", "No OS + Prior", "BOS + Prior", "COS + Prior"]
markers = ["s", "p", "P", "*", "h", "X"]
for i, file in enumerate(files):
    r_acc = np.load(path+file+"/r_acc.npy")
    try:
        r_auc = np.load(path+file+"/r_auc.npy")
        r_auc_base = np.load(path+file+"/r_auc_base.npy")
        avg_auc = np.mean(r_auc, axis=0)
        avg_auc_base = np.mean(r_auc_base, axis=0)
    except:
        r_auc = None
        avg_auc = None
        avg_auc_base = None
    plt.plot(range(r_acc.shape[-1]), np.mean(r_acc[:, 0, :], axis=0), marker=markers[i])
    avg_inc_acc = np.mean(np.mean(r_acc, axis=2), axis=0)
    avg_inc_acc_std = np.std(np.mean(r_acc, axis=2), axis=0)
    print("+================================+")
    print("| {}".format(legends[i]))
    print("+================================+")
    print("| Avg. incremental accuracy:     |")
    print("{:.4} +/- {:.3}".format(avg_inc_acc[0], avg_inc_acc_std[0]))
    if r_auc is not None:
        print("AUC:")
        print(avg_auc)
        print("AUC Baseline:")
        print(avg_auc_base)
    print("\n")
plt.grid("on")
plt.legend(legends, ncol=2)
plt.xlabel(r"Task")
plt.ylabel(r"Accuracy")
plt.tight_layout()
plt.savefig(fname=path+"ICL_results.pdf", format="pdf")
plt.show()

# Open all files and plot results: Acc. over task
path = "experiments/CIFAR100_5/Sample_Selection/"
files = ["RN32_RDB_RANDOM", "RN32_RDB_MIN", "RN32_RDB_MAX", "RN32_RDB_BALANCED"]
legends = ["Random", "Min", "Max", "Balanced"]
markers = ["s", "p", "P", "*"]
buffer_sizes = [200, 500, 1000, 2000]
for i, file in enumerate(files):
    r_acc = np.load(path+file+"/r_acc.npy")
    avg_inc_acc = np.mean(np.mean(r_acc, axis=2), axis=0)
    avg_inc_acc_std = np.std(np.mean(r_acc, axis=2), axis=0)
    plt.plot(buffer_sizes, avg_inc_acc, marker=markers[i])
    plt.fill_between(buffer_sizes, avg_inc_acc + avg_inc_acc_std, avg_inc_acc - avg_inc_acc_std, alpha=0.3)
    print("+================================+")
    print("| {}".format(legends[i]))
    print("+================================+")
    print("| Avg. incremental accuracy:     |")
    for size, acc, acc_std in zip(buffer_sizes, avg_inc_acc, avg_inc_acc_std):
        print("Buffer size {}: {:.4} +/- {:.3}".format(size, acc, acc_std))
    print("\n")
plt.grid("on")
plt.legend(legends, ncol=1)
plt.xticks(buffer_sizes)
plt.xlabel(r"Buffer Size")
plt.ylabel(r"$\mathrm{AIAcc}$")
plt.tight_layout()
plt.savefig(fname=path+"ICL_results.pdf", format="pdf")
plt.show()

# Open all files and plot results: Acc. over task for 10 steps
print("Results for 10 steps:")
path = "experiments/CIFAR100_10/"
files = ["RN32_RD", "RN32_RD_BOS", "RN32_RD_COS", "RN32_RDB", "RN32_RDB_BOS", "RN32_RDB_COS"]
legends = ["No OS", "BOS", "COS", "No OS + Prior", "BOS + Prior", "COS + Prior"]
markers = ["s", "p", "P", "*", "h", "X"]
for i, file in enumerate(files):
    r_acc = np.load(path+file+"/r_acc.npy")
    plt.plot(range(r_acc.shape[-1]), np.mean(r_acc[:, 0, :], axis=0), marker=markers[i])
    avg_inc_acc = np.mean(np.mean(r_acc, axis=2), axis=0)
    avg_inc_acc_std = np.std(np.mean(r_acc, axis=2), axis=0)
    print("+================================+")
    print("| {}".format(legends[i]))
    print("+================================+")
    print("| Avg. incremental accuracy:     |")
    print("{:.4} +/- {:.3}".format(avg_inc_acc[0], avg_inc_acc_std[0]))
    print("\n")
plt.grid("on")
plt.legend(legends, ncol=2)
plt.xlabel(r"Task")
plt.ylabel(r"Accuracy")
plt.tight_layout()
plt.savefig(fname=path+"ICL_results.pdf", format="pdf")
plt.show()

# Open all files and plot results: Acc. over task for 5 steps
print("+----------------------------+")
print("| SubImageNet                |")
print("+----------------------------+")
print("Results for 5 steps:")
path = "experiments/SubImageNet_5/"
files = ["RN18_RD", "RN18_RD_BOS", "RN18_RDB", "RN18_RDB_BOS"]
legends = ["No OS", "BOS", "No OS + Prior", "BOS + Prior"]
markers = ["s", "p", "P", "*", "h", "X"]
for i, file in enumerate(files):
    r_acc = np.load(path+file+"/r_acc.npy")
    plt.plot(range(r_acc.shape[-1]), np.mean(r_acc[:, 0, :], axis=0), marker=markers[i])
    avg_inc_acc = np.mean(np.mean(r_acc, axis=2), axis=0)
    avg_inc_acc_std = np.std(np.mean(r_acc, axis=2), axis=0)
    print("+================================+")
    print("| {}".format(legends[i]))
    print("+================================+")
    print("| Avg. incremental accuracy:     |")
    print("{:.4} +/- {:.3}".format(avg_inc_acc[0], avg_inc_acc_std[0]))
    print("\n")
plt.grid("on")
plt.legend(legends, ncol=2)
plt.xlabel(r"Task")
plt.ylabel(r"Accuracy")
plt.tight_layout()
plt.savefig(fname=path+"ICL_results.pdf", format="pdf")
plt.show()

# Open all files and plot results: Acc. over task for 10 steps
print("Results for 10 steps:")
path = "experiments/SubImageNet_10/"
files = ["RN18_RD", "RN18_RD_BOS", "RN18_RDB", "RN18_RDB_BOS"]
legends = ["No OS", "BOS", "No OS + Prior", "BOS + Prior"]
markers = ["s", "p", "P", "*", "h", "X"]
for i, file in enumerate(files):
    r_acc = np.load(path+file+"/r_acc.npy")
    plt.plot(range(r_acc.shape[-1]), np.mean(r_acc[:, 0, :], axis=0), marker=markers[i])
    avg_inc_acc = np.mean(np.mean(r_acc, axis=2), axis=0)
    avg_inc_acc_std = np.std(np.mean(r_acc, axis=2), axis=0)
    print("+================================+")
    print("| {}".format(legends[i]))
    print("+================================+")
    print("| Avg. incremental accuracy:     |")
    print("{:.4} +/- {:.3}".format(avg_inc_acc[0], avg_inc_acc_std[0]))
    print("\n")
plt.grid("on")
plt.legend(legends, ncol=2)
plt.xlabel(r"Task")
plt.ylabel(r"Accuracy")
plt.tight_layout()
plt.savefig(fname=path+"ICL_results.pdf", format="pdf")
plt.show()
