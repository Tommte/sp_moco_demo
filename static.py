import numpy as np
import matplotlib.pyplot as plt
import yaml

from misc.subsampling import choose_patterns
from misc.noise import make_noise, regularization
from misc.utils import decomp_posneg, print_metrics, select_image

import spyrit.misc.walsh_hadamard as wh


def static(Hp, Hn, f_ref):
    # synthetic data for the static problem
    M = Hp.shape[0]
    mp, mn = np.zeros(M), np.zeros(M)

    for k in range(M):
        mp[k] = np.dot(Hp[k, :], f_ref)
        mn[k] = np.dot(Hn[k, :], f_ref)

    return mp, mn


if __name__ == '__main__':
    # Read parameters file
    paramsfilename = "Params/params.yaml"
    with open(paramsfilename) as f:
        par = yaml.load(f, Loader=yaml.FullLoader)

    alpha = par['alpha']
    K = 1 / alpha
    mu_dark = par['mu_dark']
    sigma_dark = par['sigma_dark']

    n = par['n']
    img_name = par['img_name']
    image = select_image(img_name, n)

    mes_noise = "poisson"
    solver = "custom"
    type_reg = "TV-fista"

    reg = True
    rcond = 0.1
    eta = 5e-3
    n_iter = 10

    N = image.shape[0] * image.shape[1]
    M = N

    f_ref = image.reshape(N)


    ### MEAS
    Had_mat = wh.walsh2_matrix(len(image))
    subsamp = choose_patterns(M, N, "first")

    Hp, Hn = decomp_posneg(Had_mat)
    Hp, Hn = Hp[subsamp, :], Hn[subsamp, :]

    mp, mn = static(Hp, Hn, f_ref)


    ### NOISE ON MEAS
    mp, mn = make_noise(mp, alpha, K, mu_dark, sigma_dark, type=mes_noise), make_noise(mn, alpha, K, mu_dark, sigma_dark, type=mes_noise)


    ### PREPROCESSING
    # assemblage
    m = (mp - mn) / (alpha * K)


    ### RECONSTRUCTION && METRICS
    metrics = ["PSNR", "L_inf"]

    f_ref_recon = np.dot(Had_mat[subsamp, :].transpose(), m) / N

    if solver == "lstsq":
        f_regu, resi, rank, sing_val = np.linalg.lstsq(Had_mat[subsamp, :], m, rcond=rcond)

    elif solver == "custom":
        f_regu = regularization(Had_mat[subsamp, :], m, eta, type=type_reg, n_iter=n_iter)

    else:
        print("Invalid solver name")
        exit(1)

    print_metrics(metrics, f_ref_recon, f_ref, "brute")
    print_metrics(metrics, f_regu, f_ref, "regu")

    f_ref_recon = f_ref_recon.reshape(image.shape)
    f_ref = f_ref.reshape((n, n))
    f_regu = f_regu.reshape((n, n))


    ### PLOT
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))

    ax[0].imshow(f_ref, cmap="gray")
    ax[0].set_title("Image de référence")

    ax[1].imshow(m.reshape((n, n)), cmap="gray")
    ax[1].set_title("Image dans l'espace des mesures")

    if reg:
        ax[1].imshow(f_ref_recon, cmap="gray")
        ax[1].set_title("Reconstruction statique")

        ax[2].imshow(f_regu, cmap="gray")
        ax[2].set_title("Reconstruction régularisée")
    else:
        ax[2].imshow(f_ref_recon, cmap="gray")
        ax[2].set_title("Reconstruction statique")

    plt.show()
