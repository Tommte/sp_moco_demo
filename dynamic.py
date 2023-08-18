import numpy as np
import math
import time
import yaml
import matplotlib.pyplot as plt
import spyrit.misc.walsh_hadamard as wh

from skimage import transform

from misc.subsampling import choose_patterns
from misc.noise import make_noise, regularization
from misc.utils import decomp_posneg, print_metrics, normalize_img, select_image, plot_deform_img, add_colorbar
from misc.interpolation import vfast_prod_hk_U
from misc.model import deform_u, deform_v


def dynamic_f0_f1_f2(x1, x2, Hp, Hn, deformation, f_ref, T, forme_interp):
    M, N = Hp.shape
    n = int(math.sqrt(N))
    mp_u, mn_u = np.zeros(M), np.zeros(M)
    mp_v, mn_v = np.zeros(M), np.zeros(M)
    H_d1 = np.zeros(shape=(M, N))
    H_d2 = np.zeros(shape=(M, N))

    order = 1
    if forme_interp == "doors":
        order = 0

    for k in range(M):
        t = k * T / (M - 1)

        x11, x22 = deform_v(deformation, t, T, x1, x2, M)

        grid = n * np.array([x22, x11]).reshape((2, n, n))
        imt = transform.warp(f_ref.reshape((n, n)), grid, order=order)

        imt = imt.reshape((N,))

        mp_u[k] = np.dot(Hp[k, :], imt)
        mn_u[k] = np.dot(Hn[k, :], imt)

        ## PLOT DEFORMATION % t :
        time_step, pause = n, 0.01
        plot_deform_img(imt, k, time_step, t, pause)


        ### F1 Lagrangian
        H_d1_p = vfast_prod_hk_U(Hp[k, :], x11, x22, forme_interp)
        H_d1_n = vfast_prod_hk_U(Hn[k, :], x11, x22, forme_interp)

        H_d1[k, :] = H_d1_p.reshape(N) - H_d1_n.reshape(N)


        ### F2 Eulerian
        x11, x22 = deform_u(deformation, t, T, x1, x2, M)
        det_v = 1  # affine motion

        grid = n * np.array([x22, x11]).reshape((2, n, n))
        motifp_tform = abs(det_v) * transform.warp(Hp[k, :].reshape((n, n)), grid, order=order)
        motifn_tform = abs(det_v) * transform.warp(Hn[k, :].reshape((n, n)), grid, order=order)

        motifp_tform, motifn_tform = motifp_tform.reshape((N,)), motifn_tform.reshape((N,))

        H_d2[k, :] = motifp_tform - motifn_tform
        mp_v[k] = np.dot(motifp_tform, f_ref)
        mn_v[k] = np.dot(motifn_tform, f_ref)

    return mp_u, mn_u, H_d1, mp_v, mn_v, H_d2


if __name__ == '__main__':
    # Read parameters file
    paramsfilename = "Params/params.yaml"
    with open(paramsfilename) as f:
        par = yaml.load(f, Loader=yaml.FullLoader)

    alpha = par['alpha']
    K = 1 / alpha
    mu_dark = par['mu_dark']
    sigma_dark = par['sigma_dark']

    start = time.time()

    n = par['n']
    img_name = par['img_name']
    f_ref = select_image(img_name, n)

    N = f_ref.shape[0] * f_ref.shape[1]
    M = N

    f_ref = f_ref.reshape(N)
    f_ref = normalize_img([f_ref])[0]

    nb_periods = par['nb_periods']
    T = nb_periods * par['T_period']

    interval = np.linspace(0, n - 1, n) / n
    x1, x2 = np.meshgrid(interval, interval)
    x1.resize(N)
    x2.resize(N)

    forme_interp = "bilinear"

    mes_noise = "no-noise"
    param_noise = False  # bruit sur les params

    solver = "custom"
    type_reg = "TV-fista"

    rcond = 1e-15
    # rcond = 0.1
    list_eta = [1e-2]
    n_iter = 10  # n_iter for reg

    deformation = par['deformation']
    deformation_noisy = par['deformation_noisy']

    read_subsamp = False
    save_subsamp = False


    ### BUILD HADAMARD MATRIX
    Had_mat = wh.walsh2_matrix(n)

    if read_subsamp:
        subsamp = np.load("./sim_data/subsample.npy")
    else:
        subsamp = choose_patterns(M, N, "first")

    if save_subsamp:
        np.save("./sim_data/subsample.npy", subsamp)

    Hp, Hn = decomp_posneg(Had_mat)
    Hp, Hn = Hp[subsamp, :], Hn[subsamp, :]

    print("-------- DYNAMIC ---------")
    print("Interpolation :", forme_interp)
    print("Bruit :", mes_noise, "sur mesures  || ", param_noise, "sur mouvement")
    print("Solveur :", solver)
    print("--------------------------\n")

    t_start = time.time()
    print("Temps jusqu'à présent : ", t_start - start, "s")


    ### SIMULATION OF MEAS AND DYNAMIC MATRIX CONSTRUCTION
    print("-------- F0 & F1 & F2 --------")
    mp_u_nn, mn_u_nn, H_d1, mp_v_nn, mn_v_nn, H_d2 = dynamic_f0_f1_f2(x1, x2, Hp, Hn, deformation, f_ref, T, forme_interp)

    if param_noise:
        _, _, H_d1, _, _, H_d2 = dynamic_f0_f1_f2(x1, x2, Hp, Hn, deformation_noisy, f_ref, T, forme_interp)

    print("Done")

    t_f0f1f2 = time.time()
    print("Temps pour F0 & F1 & F2: ", t_f0f1f2 - t_start, "s")

    plt.close()

    ### ADD NOISE ON MEAS
    mp_u, mn_u = make_noise(mp_u_nn, alpha, K, mu_dark, sigma_dark, type=mes_noise), make_noise(mn_u_nn, alpha, K, mu_dark, sigma_dark, type=mes_noise)
    mp_v, mn_v = make_noise(mp_v_nn, alpha, K, mu_dark, sigma_dark, type=mes_noise), make_noise(mn_v_nn, alpha, K, mu_dark, sigma_dark, type=mes_noise)

    # assemblage
    m_u = (mp_u - mn_u) / (alpha * K)
    m_v = (mp_v - mn_v) / (alpha * K)

    for eta in list_eta:
        ### RECONSTRUCTION && METRICS
        f_ref = f_ref.reshape((N,))
        plt.figure()
        print("LAMBDA = ", eta)

        f_f0_u = np.dot(Had_mat[subsamp, :].transpose(), m_u) / N
        f_f0_v = np.dot(Had_mat[subsamp, :].transpose(), m_v) / N

        f_f1_u = np.zeros(N)
        f_f1_v = np.zeros(N)
        f_f2_u = np.zeros(N)
        f_f2_v = np.zeros(N)

        if solver == "custom":
            # f_f2_v = regularization(H_d2, m_v, eta, type=type_reg, n_iter=n_iter)
            f_f1_u = regularization(H_d1, m_u, eta, type=type_reg, n_iter=n_iter)

            f_f2_u = regularization(H_d2, m_u, eta, type=type_reg, n_iter=n_iter)
            # f_f1_v = regularization(H_d1, m_v, eta, type=type_reg, n_iter=n_iter)

        elif solver == "lstsq":
            # f_f2_v, resi, rank, sing_val_d1 = np.linalg.lstsq(H_d2, m_v, rcond=rcond)
            f_f1_u, resi2, rank2, sing_val_d2 = np.linalg.lstsq(H_d1, m_u, rcond=rcond)

            f_f2_u, resi, rank, sing_val_d1 = np.linalg.lstsq(H_d2, m_u, rcond=rcond)
            # f_f1_v, resi2, rank2, sing_val_d2 = np.linalg.lstsq(H_d1, m_v, rcond=rcond)

            # from utils import plot_singular_values
            # plot_singular_values(sing_val_d1, sing_val_d2, rcond)

        elif solver == "adjoint":
            f_f1_u = np.dot(H_d1.transpose(), m_u) / N
            f_f2_u = np.dot(H_d2.transpose(), m_u) / N

        else:
            print("Choose a correct solver name")
            exit(1)

        metrics = ["PSNR", "L_inf"]

        print_metrics(metrics, f_f0_u, f_ref, "F0 m_u")
        # print_metrics(metrics, f_f0_v, f_ref, 'F0 m_v')

        print_metrics(metrics, f_f1_u, f_ref, "F1 m_u")
        # print_metrics(metrics, f_f1_v, f_ref, "F1 m_v")

        print_metrics(metrics, f_f2_u, f_ref, 'F2 m_u')
        # print_metrics(metrics, f_f2_v, f_ref, "F2 m_v")

        f_ref = f_ref.reshape((n, n))
        f_f0_u = f_f0_u.reshape(f_ref.shape)
        f_f0_v = f_f0_v.reshape(f_ref.shape)
        f_f2_u = f_f2_u.reshape(f_ref.shape)
        f_f2_v = f_f2_v.reshape(f_ref.shape)
        f_f1_u = f_f1_u.reshape(f_ref.shape)
        f_f1_v = f_f1_v.reshape(f_ref.shape)

        end = time.time()
        print("Temps d'éxécution total : ", end - start, "s")


        ### PLOT
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))

        im_0 = ax[0].imshow(f_f0_u, cmap="gray")
        ax[0].set_title("F0 m_d1")
        add_colorbar(fig, im_0, ax[0])

        im_1 = ax[1].imshow(f_f1_u, cmap="gray")
        ax[1].set_title("F1 m_d1")
        add_colorbar(fig, im_1, ax[1])

        im_2 = ax[2].imshow(f_f2_u, cmap="gray")
        ax[2].set_title("F2 m_d1")
        add_colorbar(fig, im_2, ax[2])

        plt.suptitle("Reconstructions dynamiques", fontsize=16)

        plt.tight_layout()

    plt.show()
