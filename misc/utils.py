import numpy as np
import math
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom, camera
from skimage import transform
from mpl_toolkits.axes_grid1 import make_axes_locatable


def decomp_posneg(H):
    # Splits a Hadamard matrix into its positive and negative parts (slow)
    Hp, Hn = np.zeros(shape=H.shape), np.zeros(shape=H.shape)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i][j] > 0:
                Hp[i][j] = 1
            else:
                Hn[i][j] = 1

    return Hp, Hn

def print_metrics(metrics, recon, img, pb=""):
    print("Metrics ", pb)

    recon, img = normalize_img([recon, img])

    for met in metrics:
        if met == "PSNR":
            PSNR = get_PSNR(recon, img)
            print("PSNR = ", PSNR)

        elif met == "L_inf":
            err_inf = np.abs(img - recon).max()
            print("erreur L_inf = ", err_inf)

        elif met == "L_2":
            err_L2 = np.linalg.norm(img - recon)
            print("erreur L_2 = ", err_L2)

    print("-----------------------")


def get_PSNR(recon, img):
    N = img.size
    diff = recon - img
    MSE = 1. / N * np.dot(diff.transpose(), diff)
    PSNR = 10 * math.log10(2 ** 2 / MSE)

    return PSNR

def normalize_img(img_list):
    img_norm_list = []
    for img in img_list:
        img = (img - img.min()) / (img.max() - img.min())
        img_norm_list.append(img)

    return img_norm_list


def select_image(name, n):
    if name == "shepp-logan":
        image = shepp_logan_phantom()
        image = transform.resize(image, (n, n))

    elif name == "small-shepp-logan":
        image_petit = shepp_logan_phantom()
        image_petit = transform.resize(image_petit, (n//2, n//2))
        image = np.zeros((n, n))
        image[int(n/4):int(3*n/4), int(n/4):int(3*n/4) ] = image_petit

    elif name == "white_spot":
        image = np.zeros((n, n))
        c = [n//2, n//2]
        # c = [7 * n // 8, n // 8]
        for l in range(n):
            for k in range(n):
                if (k - c[0]) ** 2 + (l - c[1]) ** 2 < (n // 10) ** 2:
                    image[l, k] = 1

    else:
        print("Nom d'image non valide")
        exit(1)

    image = normalize_img([image])[0]

    return image

def plot_deform_field(dx, dy, x1, x2, i, time_step, pause, message=None, axes=None, fig=None):
    if i % time_step == 0:
        N = x1.size
        n = int(math.sqrt(N))

        if axes is None:
            plt.clf()
            color = np.sqrt(dx**2 + dy**2)
            plt.quiver(x1, -x2, dx, dy, color, angles="xy", scale_units='xy', scale=1)

        else:
            axes[0].clear()
            axes[1].clear()

            cs1 = axes[0].contourf(dx.reshape(n, n), cmap='Spectral', vmin=-0.01, vmax=0.02, origin='image')#, levels=40)
            cs2 = axes[1].contourf(dy.reshape(n, n), cmap='Spectral', vmin=-0.02, vmax=0.03, origin='image')#, levels=40)

            # fig.colorbar(cs1, ax=axes[0], shrink=0.9)
            # fig.colorbar(cs2, ax=axes[1], shrink=0.9)

            axes[0].set_title("Déformation suivant x")
            axes[1].set_title("Déformation suivant y")
            # plt.colorbar()

        if message is None:
            message = "Champ de déformation"
        plt.suptitle(message, fontsize=16)
        plt.pause(pause)

def plot_deform_img(img, i, time_step, t, pause):
    if i % time_step == 0:
        N = img.size
        n = int(math.sqrt(N))
        img = img.reshape((n, n))
        plt.clf()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.suptitle("temps t = %.2f " % t, fontsize=14)
        plt.pause(pause)

def plot_diag_mat(H_d1, H_d2):
    plt.clf()
    plt.plot(np.diag(np.dot(H_d1, H_d1.transpose())))
    plt.title("Diagonale de H_d1 H_d1^T")
    plt.figure()

    plt.plot(np.diag(np.dot(H_d2, H_d2.transpose())))
    plt.title("Diagonale de H_d2 H_d2^T")
    plt.figure()

    plt.clf()
    plt.plot(np.diag(np.dot(H_d1, H_d1.transpose())) - np.diag(np.dot(H_d2, H_d2.transpose())))
    plt.title("Diff diag")
    plt.show()

def plot_meas(m_u, m_v):
    # Pour visualiser les différences entre m_u et m_v
    print("Erreur m_u % m_v = ", np.linalg.norm(m_u - m_v))
    print(np.linalg.norm(m_v))
    print(np.linalg.norm(m_u))

    plt.clf()
    plt.plot(m_v, "+-", color='blue')
    plt.plot(m_u, "+-", color='orange')
    plt.suptitle("Mesures m_u (orange) et m_v (bleu)", fontsize=14)

    plt.figure()
    plt.plot(m_u - m_v, "+-", color="red")
    plt.suptitle("Différence m_u - m_v", fontsize=14)
    plt.show()

def plot_singular_values(sing_val_d1, sing_val_d2, rcond):
    print("nb singular value = ", sing_val_d1.size, " & ", sing_val_d2.size)
    plt.clf()
    plt.plot(sing_val_d2, color='blue', label="valeurs singulières")
    plt.plot(rcond * sing_val_d2.max() * np.ones(sing_val_d2.shape), color='red', label='seuil rcond')
    plt.title("Valeurs singulières de H_d2")
    plt.legend()
    plt.figure()

    plt.clf()
    plt.plot(sing_val_d1, color='blue', label="valeurs singulières")
    plt.plot(rcond * sing_val_d1.max() * np.ones(sing_val_d1.shape), color='red', label='seuil rcond')
    plt.title("Valeurs singulières de H_d1")
    plt.legend()
    plt.figure()

    plt.clf()
    plt.plot(sing_val_d2 - sing_val_d1, color='blue', label="valeurs singulières")
    plt.title("Différence des valeurs singulières de H_d1 et H_d2")
    plt.legend()
    plt.show()


def add_colorbar(fig, im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

