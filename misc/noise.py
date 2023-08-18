import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import denoise_tv_chambolle


def make_noise(m, alpha, K, mu_dark=0, sigma_dark=1, type="no-noise"):
    if type == "no-noise":
        mes = K * alpha * m

    elif type == "poisson":
        mes = K * np.random.poisson(alpha * m)

    elif type == "gaussian":
        eps = np.random.normal(mu_dark, sigma_dark, m.size)
        mes = (1 + eps) * m

    elif type == "mixed":
        mes = K * np.random.poisson(alpha * m) + np.random.normal(mu_dark, sigma_dark ** 2, m.size)

    else:
        print("Type of noise not implemented")
        exit(1)

    return mes


def regularization(A, b, eta, type="TV-fista", n_iter=10):
    M, N = A.shape
    n = int(math.sqrt(N))

    np.random.seed(1)

    if type == "TV-ista":
        f = np.zeros(N)
        tau = 0.5 * 1. / N
        loss = []

        for k in range(n_iter):
            # print(k)
            f = f - tau * np.dot(A.transpose(), np.dot(A, f) - b) #- f * 0.1 * tau
            f_square = f.reshape((n, n))
            f_square = denoise_tv_chambolle(f_square, weight=tau / eta, eps=2e-5, max_num_iter=2000)
            f = f_square.reshape((N,))

            # if k % 10 == 0:
            # plt.clf()
            # plt.imshow(f.reshape((n, n)), cmap="gray")
            # plt.suptitle("ISTA " + str(k))
            # plt.pause(0.2)

            # loss.append(0.5 * eta * np.linalg.norm(np.dot(A, f) - b)**2 + np.linalg.norm(calc_gradient(f), ord=1))

        # np.save("./sim_data/loss/ista_loss.npy", loss)

        # plt.clf()
        # plt.plot(loss, "o--")
        # plt.title("Loss")
        # plt.show()

    elif type == "TV-fista":
        x = np.zeros(N)
        y = np.zeros(N)
        tau = 0.5 * 1. / N
        t_k = 1
        # a = 4
        loss = []

        for k in range(n_iter):
            x_km1 = np.copy(x)

            y = y - tau * np.dot(A.transpose(), np.dot(A, y) - b) #- y * 0.1 * tau
            y_square = y.reshape((n, n))
            y_square = denoise_tv_chambolle(y_square, weight=tau / eta, eps=2e-5, max_num_iter=2000)
            # y_square = denoise_tv_bregman(y_square, weight=eta/tau, max_num_iter=100, eps=1e-3, isotropic=True)
            x = y_square.reshape((N,))

            t_kp1 = (1 + math.sqrt(1 + 4 * t_k**2))/2
            # t_kp1 = (k+1 + a + 1) / a  # Chambolle's idea
            y = x + (t_k - 1) / t_kp1 * (x - x_km1)

            t_k = t_kp1

            # if k % 10 == 0:
            # plt.clf()
            # plt.imshow(y.reshape((n, n)), cmap="gray")
            # plt.suptitle("FISTA itération " + str(k))
            # plt.pause(0.2)

            # loss.append(0.5 * eta * np.linalg.norm(np.dot(A, x) - b)**2 + np.linalg.norm(np.dot(A.transpose(), np.dot(A, x) - b), ord=1)) #faux
            # loss.append(0.5 * eta * np.linalg.norm(np.dot(A, x) - b)**2 + np.linalg.norm(calc_gradient(x), ord=1))

        f = x.copy()

        # plt.clf()
        # plt.plot(loss, "o--")
        # plt.title("Loss")
        # plt.show()

    elif type == "L2":
        f = np.zeros(N)
        tau = 0.5 * 1. / N
        sigma_inv = np.linalg.inv(np.load("./stats/Cov_64x64.npy"))
        loss = []

        for k in range(n_iter):
            f = f - tau * np.dot(A.transpose(), np.dot(A, f) - b) - 2 * eta * tau * f  # L2
            # f = f - tau * np.dot(A.transpose(), np.dot(A, f) - b) - 2 * eta * tau * np.dot(sigma_inv, f)  # L2 généralisée

            # if k % 10 == 0:
            plt.clf()
            plt.imshow(f.reshape((n, n)), cmap="gray")
            plt.suptitle("ISTA " + str(k))
            plt.pause(0.01)

            loss.append(0.5 * np.linalg.norm(np.dot(A, f) - b) ** 2 + eta * np.dot(f.transpose(),  f))
            # loss.append(0.5 * np.linalg.norm(np.dot(A, f) - b) ** 2 + eta * np.dot(f.transpose(), np.dot(sigma_inv, f)))

        plt.clf()
        plt.plot(loss, "o--")
        plt.title("Loss")
        plt.show()

    else:
        print("Type de régularisation non implémentée")
        exit(1)

    return f


