import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

import spyrit.misc.walsh_hadamard as wh


def choose_patterns(M, N, method):
    if (method == "first"):
        subsamp = np.arange(M)

    elif (method == "random"):
        subsamp = np.arange(N)
        np.random.shuffle(subsamp)

    else:
        print("Choisir une méthode de subsampling valide")
        exit(1)

    return subsamp[0:M]


def low_freq_2D(Had_mat):
    # Fonction pour vérifier que l'ordre de Walsh était bien ordonné des basses fréquences aux hautes fréquences
    N = Had_mat.shape[0]
    n = int(math.sqrt(N))
    H = Had_mat.reshape(N, n, n)
    m = n // 4
    H_lf = np.zeros((m*n, N))
    for i in range(N):
        subsamp = choose_patterns(m, n, "first")
        H_lf[:, i] = H[i, subsamp, :].reshape((m*n,))

    return H_lf


def show_patterns(Had_mat):
    # Plot the patterns form the forward matrix in the form of images
    N = Had_mat.shape[0]
    n = int(math.sqrt(N))

    P = np.zeros((n, n, N))

    for i in range(N):
        P[:, :, i] = Had_mat[i, :].reshape((n, n))

    fig, ax = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                ax[i][j].imshow(P[:, :, i * n + j], cmap="binary")
            else:
                ax[i][j].imshow(P[:, :, i*n + j], cmap="gray")

    plt.tight_layout()
    plt.show()

    return P


if __name__ == '__main__':
    ## Verif walsh order
    Had = wh.walsh2_matrix(4)  # walsh order 2D

    h_nat = hadamard(16)  # natural order 1D (sylvester)
    subsamp = wh.sequency_perm_ind(16)
    h_walsh_scipy_1D = h_nat[subsamp, :]

    h_walsh_spyrit_1D = wh.walsh_matrix(16)  # walsh order 1D

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].matshow(h_walsh_scipy_1D)
    ax[0].set_title("Scipy reordered with spyrit", fontsize=16)

    ax[1].matshow(h_walsh_spyrit_1D)
    ax[1].set_title("Spyrit", fontsize=16)

    plt.suptitle("1D Hadamard matrices in Walsh order", fontsize=16)
    plt.show()


    ## Verif subsampling order in 2D
    H_lf = low_freq_2D(Had)
    subsamp_lf = choose_patterns(4, 16, "first")

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].matshow(Had[subsamp_lf, :])
    ax[0].set_title("First patterns are retained", fontsize=16)

    ax[1].matshow(H_lf)
    ax[1].set_title("Low frequencies manually chosen", fontsize=16)

    plt.suptitle("Subsampled 2D Hadamard matrices", fontsize=16)
    plt.show()


    ## Plot patterns
    patterns = show_patterns(Had)
