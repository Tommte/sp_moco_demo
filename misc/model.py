import numpy as np
import math
import yaml
import matplotlib.pyplot as plt

from skimage import transform

from misc.utils import normalize_img, select_image, plot_deform_img


# LOAD PARAMS AS GLOBAL PARAMETERS
paramsfilename = "Params/params.yaml"
with open(paramsfilename) as f:
    par = yaml.load(f, Loader=yaml.FullLoader)

def s(t):
    return 0.05 * np.cos(2 * math.pi * t / par['T_period']) + 0.95

def s2(t):
    return 0.03 * np.cos(2 * math.pi * t / par['T_period']) + 0.97

def s3(t):
    times = np.array([0, 275, 430, 500, 750, 1000])  # ms
    P = (np.array([2050, 3400, 2700, 3000, 2250, 2050]) / 3400 + 3) / 4  # mmHg normalisés

    time = t % par['T_period']

    for i in range(P.size - 1):
        if times[i] <= time <= times[i + 1]:
            return P[i] + (P[i + 1] - P[i]) * (time - times[i]) / (times[i + 1] - times[i])

    print("Error, time is not in [0, T]")
    exit(1)

def s4(t):
    # errors on params from s3
    times = np.array([0, 275, 430, 500, 750, 1000])  # ms
    P = (np.array([2150, 3300, 2600, 3100, 2250, 2150]) / 3400 + 3) / 4  # mmHg normalisés

    time = t % par['T_period']

    for i in range(P.size):
        if times[i] <= time < times[i + 1]:
            return P[i] + (P[i + 1] - P[i]) * (time - times[i]) / (times[i + 1] - times[i])

    print("Error, time is not in [0, T]")
    exit(1)


def deform_v(deformation, t, T, x1, x2, M):
    # déformation v (sens inverse, déformation inverse de u). On déforme la grille (x1, x2) selon la deformation "deformation"
    c = 0.5 * np.array([1, 1])
    N = x1.size
    n = int(math.sqrt(N))

    if deformation == 0:
        # deformation papier hahn (x1 et x2 changés, convention article différent?) #ancien 9
        x11 = c[0] + s(t) * (x1 - c[0])
        x22 = c[1] + (x2 - c[1])/s(t)

    elif deformation == 0.5:
        # on change directement les paramètres de la déformation 0 pour modéliser une mauvaise estimation de mouvement
        x11 = c[0] + s2(t) * (x1 - c[0])
        x22 = c[1] + (x2 - c[1]) / s2(t)

    elif deformation == 1:
        # translation diagonal 1
        theta = - 3 * math.pi / 4
        n_pix = 18 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 + d * math.cos(theta) * t / T
        x22 = x2 + d * math.sin(theta) * t / T

    elif deformation == 1.5:
        # translation noisy
        theta = - 3 * math.pi / 4
        n_pix = 16 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 + d * math.cos(theta) * t / T
        x22 = x2 + d * math.sin(theta) * t / T

    elif deformation == 2:
        # translation diagonal 2
        theta = - math.pi / 4
        n_pix = 18 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 + d * math.cos(theta) * t / T
        x22 = x2 + d * math.sin(theta) * t / T

    elif deformation == 3:
        # rotation
        theta = 2 * math.pi/3 * t / T
        a = 1 * np.array([1, 1])

        x11 = c[0] + a[0] * math.cos(theta) * (x1 - c[0]) - a[1] * math.sin(theta) * (x2 - c[1])
        x22 = c[1] + a[0] * math.sin(theta) * (x1 - c[0]) + a[1] * math.cos(theta) * (x2 - c[1])

    elif deformation == 4:
        # mouvement du piston
        x11 = c[0] + s3(t) * (x1 - c[0])
        x22 = c[1] + (x2 - c[1]) / s3(t)

    elif deformation == 4.5:
        # mouvement du piston à params perturbés
        x11 = c[0] + s4(t) * (x1 - c[0])
        x22 = c[1] + (x2 - c[1]) / s4(t)

    else:
        print("Deformation non implémentée")
        exit(1)

    return x11, x22


def deform_u(deformation, t, T, x1, x2, M):
    # déformation u (sens direct). On déforme la grille (x1, x2) selon la deformation "deformation"
    N = x1.size
    n = int(math.sqrt(N))
    c = 0.5 * np.array([1, 1])

    if deformation == 0:
        # deformation papier hahn (x1 et x2 changés, convention article différent?)
        x11 = c[0] + (x1 - c[0]) / s(t)
        x22 = c[1] + s(t) * (x2 - c[1]) #+ 0.44 * (1 - s(t))

    elif deformation == 0.5:
        # on change directement les paramètres de la déformation 0 pour modéliser une mauvaise estimation de mouvement
        x22 = c[0] + s2(t) * (x2 - c[0])
        x11 = c[1] + (x1 - c[1]) / s2(t)

    elif deformation == 1:
        # translation on diagonal 1
        theta = - 3 * math.pi / 4
        n_pix = 18 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 - d * math.cos(theta) * t / T
        x22 = x2 - d * math.sin(theta) * t / T

    elif deformation == 1.5:
        # translation noisy
        theta = - 3 * math.pi / 4
        n_pix = 16 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 - d * math.cos(theta) * t / T
        x22 = x2 - d * math.sin(theta) * t / T

    elif deformation == 2:
        # translation on diagonal 2
        theta = - math.pi / 4
        n_pix = 18 * M / N
        d = np.sqrt(2) * n_pix / n
        x11 = x1 - d * math.cos(theta) * t / T
        x22 = x2 - d * math.sin(theta) * t / T

    elif deformation == 3:
        # rotation
        theta = -2 * math.pi/3 * t / T
        a = 1 * np.array([1, 1])

        x11 = c[0] + a[0] * math.cos(theta) * (x1 - c[0]) - a[1] * math.sin(theta) * (x2 - c[1])
        x22 = c[1] + a[0] * math.sin(theta) * (x1 - c[0]) + a[1] * math.cos(theta) * (x2 - c[1])

        det = np.ones(shape=(n, n)) * a[0] * a[1]

    elif deformation == 4:
        x22 = c[0] + s3(t) * (x2 - c[0])
        x11 = c[1] + (x1 - c[1]) / s3(t)

    elif deformation == 4.5:
        x22 = c[0] + s4(t) * (x2 - c[0])
        x11 = c[1] + (x1 - c[1]) / s4(t)

    else:
        print("Deformation non implémentée")
        exit(1)

    return x11, x22



if __name__ == '__main__':
    dim = 2
    n = par['n']

    nb_periods = par['nb_periods']
    T = nb_periods * par['T_period']  # [T] = ms

    deformation = par['deformation']

    filenames = []

    ### ------------------ 1D --------------------
    if dim == 1:
        x = np.linspace(0, n - 1, n) / n

        M = n
        y = np.zeros(M)

        c = 0.5
        s_arr = np.zeros(M)

        for k in range(M):
            t = k * T / (M - 1)
            if deformation == 0:
                y = c + s(t) * (x - c)
                s_arr[k] = s(t)
            elif deformation == 4:
                y = c + s3(t) * (x - c)
                s_arr[k] = s3(t)

            # plot deformation
            # plt.clf()
            plt.plot(x, y)
            plt.title("t = %.2f" % t)
            # plt.ylim([0, 1])
            plt.pause(0.2)


        # Plot s(t)
        time_arr = np.array([k * T / (M-1) for k in range(M)])
        # s_arr = s(time_arr)
        plt.clf()
        plt.plot(time_arr, s_arr, "+--")
        plt.title("s(t)")
        plt.show()


    ### ------------------ 2D --------------------
    elif dim == 2:
        img_name = par['img_name']
        image = select_image(img_name, n)

        N = image.shape[0] * image.shape[1]
        M = N

        f_ref = image.reshape(N)
        f_ref = normalize_img([f_ref])[0]

        interval = np.linspace(0, n - 1, n) / n
        x1, x2 = np.meshgrid(interval, interval)
        x1.resize(N)
        x2.resize(N)

        order = 1
        plt.figure(figsize=(9, 9))
        x11_old, x22_old = x1.copy(), x2.copy()

        for k in range(M):
            t = k * T / (M - 1)

            x11, x22 = deform_v(deformation, t, T, x1, x2, M)

            grid = n * np.array([x22, x11]).reshape((2, n, n))
            imt = transform.warp(f_ref.reshape((n, n)), grid, order=order)
            imt = imt.reshape((N,))

            ## PLOT DEFORMATION % t :
            time_step, pause = n, 0.01
            plot_deform_img(imt, k, time_step, t, pause)

        #     if k % time_step == 0:
        #         filename = f'gif/sim_deformation_1/-{k}.png'
        #         filenames.append(filename)
        #
        #         # save frame
        #         plt.savefig(filename)
        #         plt.close()
        #
        # np.save("gif/sim_deformation_1/filenames.npy", filenames)

        plt.close()

    else:
        print("Choose between dim 1 or 2")
        exit(1)
