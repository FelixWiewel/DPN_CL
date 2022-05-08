"""
This file is used for illustrating the effect of rehearsal batch size.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{amsmath}"})

# Parameters
tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

# Class means
beta_bu = np.zeros(10)

# Create sub plots
fig, ax = plt.subplots(4, 5, sharey=True, sharex=True)

for i, t in enumerate(tasks):
    beta_cu = np.zeros(10)
    beta_cu[t] = 1.0

    # Plot class means
    ax[0, i].bar(np.concatenate(tasks, axis=0), beta_cu/np.sum(beta_cu), color=colors[0])
    if i != 0:
        ax[1, i].bar(np.concatenate(tasks, axis=0), beta_bu/np.sum(beta_bu), color=colors[1])
    # Equal rehearsal and new data
    if i == 0:
        ax[2, i].bar(np.concatenate(tasks, axis=0), beta_cu/np.sum(beta_cu), color=colors[0])
    else:
        ax[2, i].bar(np.concatenate(tasks, axis=0), 0.5*beta_cu/np.sum(beta_cu), color=colors[0])
        ax[2, i].bar(np.concatenate(tasks, axis=0), 0.5*beta_bu/np.sum(beta_bu), color=colors[1])
    # True
    ax[3, i].bar(np.concatenate(tasks, axis=0), (beta_cu + beta_bu)/np.sum(beta_cu + beta_bu), color=colors[2])

    # Add labels
    ax[0, 0].set_ylabel(r"a)", rotation="horizontal")
    ax[1, 0].set_ylabel(r"b)", rotation="horizontal")
    ax[2, 0].set_ylabel(r"c)", rotation="horizontal")
    ax[3, 0].set_ylabel(r"d)", rotation="horizontal")

    beta_bu[t] = 1.0

for i in range(4):
    for j in range(5):
        ax[0, j].set_title(r"$\mathcal{{T}}_{}$".format(j+1))
        ax[i, j].grid(False)
ax[3, 2].set_xlabel(r"Class means $\boldsymbol{\mu}$")
plt.savefig("../../Paper/Conference-LaTeX-template_10-17-19/Figures/rehearsal_class_means.pdf", format="pdf")
plt.show()