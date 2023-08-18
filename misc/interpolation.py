import numpy as np
import math

def beta_1D(x, forme, n):
    # x in [-1/n , 1/n]
    if forme == "lineaire":
        if -1./n < x <= 0:
            return n * (x + 1./n)

        elif 0 < x <= 1./n:
            return n * (1./n - x)

        return 0

    else:
        print("interp 1D pas implémentée")
        exit(1)


def beta(x, y, forme, n):
    if forme == "porte":
        if -0.5/n < x <= 0.5/n:
            if -0.5/n < y <= 0.5/n:
                return 1
        return 0

    elif forme == "bi-lineaire-faux":
        res = beta_1D(x, "lineaire", n) * beta_1D(y, "lineaire", n)
        return res

    else:
        print("interp 2D pas implémentée")
        exit(1)

def coefs_interp(x, y, x1, x2, forme, n, indices):
    # returns array with coef of [(x_{i-1}, y_j), (x_i, y_j), (x_{i+1}, y_j), (x_i, y_{j-1}), (x_i, y_{j+1})]
    coefs = np.zeros(5)
    i_x, i_y = indices

    if forme == "porte":
        x, y = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        assert -0.5 / n <= x <= 0.5 / n, f" x = {x}"
        assert -0.5 / n <= y <= 0.5 / n, f" y = {y}"

        coefs[1] = 1

    elif forme == "triangle":
        x, y = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        assert -0.5 / n <= x <= 0.5 / n, f" x = {x}"
        assert -0.5 / n <= y <= 0.5 / n, f" y = {y}"

        coefs[1] = 2
        if x < 0:
            if i_x > 0:
                coefs[0] += n * x
                coefs[1] -= n * x
        else:
            if i_x < n - 1:
                coefs[1] += n * x
                coefs[2] -= n * x

        if y < 0:
            if i_y > 0:
                coefs[1] -= n * y
                coefs[3] += n * y
        else:
            if i_y < n - 1:
                coefs[1] += n * y
                coefs[4] -= n * y

        coefs *= 0.5

    elif forme == "bi-lineaire-faux":
        x_0, y_0 = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        coefs[1] = beta(x_0, y_0, "bi-lineaire-faux", n)

        if i_x < n - 1:
            x_1, y_1 = x - x1[i_x + 1 + n * i_y], y - x2[i_x + 1 + n * i_y]
            coefs[2] = beta(x_1, y_1, "bi-lineaire-faux", n)

        if i_y < n - 1:
            x_2, y_2 = x - x1[i_x + n * (i_y + 1)], y - x2[i_x + n * (i_y + 1)]
            coefs[4] = beta(x_2, y_2, "bi-lineaire-faux", n)

        if i_y > 0:
            x_3, y_3 = x - x1[i_x + n * (i_y - 1)], y - x2[i_x + n * (i_y - 1)]
            coefs[3] = beta(x_3, y_3, "bi-lineaire-faux", n)

        if i_x > 0:
            x_4, y_4 = x - x1[i_x - 1 + n * i_y], y - x2[i_x - 1 + n * i_y]
            coefs[0] = beta(x_4, y_4, "bi-lineaire-faux", n)

    else:
        print("Nom d'interpolation non valide")
        exit(1)

    return coefs

def interp(x, y, x1, x2, forme, n, indices, img_ref):
    i_x, i_y = indices

    if forme == "porte":
        x, y = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        assert -0.5 / n <= x <= 0.5 / n, f" x = {x}"
        assert -0.5 / n <= y <= 0.5 / n, f" y = {y}"

        return img_ref[i_x + n*i_y]

    elif forme == "triangle":
        x, y = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        assert -0.5 / n <= x <= 0.5 / n, f" x = {x}"
        assert -0.5 / n <= y <= 0.5 / n, f" y = {y}"

        if x < 0:
            if i_x > 0:
                px = img_ref[i_x + n*i_y] + (img_ref[i_x - 1 + n*i_y] - img_ref[i_x + n*i_y]) * n * x
            else:
                px = img_ref[i_x + n * i_y]
        else:
            if i_x < n - 1:
                px = img_ref[i_x + n*i_y] - (img_ref[i_x + 1 + n * i_y] - img_ref[i_x + n * i_y]) * n * x
            else:
                px = img_ref[i_x + n * i_y]

        if y < 0:
            if i_y > 0:
                py = img_ref[i_x + n*i_y] + (img_ref[i_x + n*(i_y - 1)] - img_ref[i_x + n*i_y]) * n * y
            else:
                py = img_ref[i_x + n * i_y]
        else:
            if i_y < n - 1:
                py = img_ref[i_x + n*i_y] - (img_ref[i_x + n * (i_y + 1)] - img_ref[i_x + n * i_y]) * n * y
            else:
                py = img_ref[i_x + n * i_y]

        return (px + py)/2

    elif forme == "bi-lineaire-faux":
        x_0, y_0 = x - x1[i_x + n * i_y], y - x2[i_x + n * i_y]
        res = beta(x_0, y_0, "bi-lineaire-faux", n) * img_ref[i_x + n * i_y]

        if i_x < n - 1:
            x_1, y_1 = x - x1[i_x + 1 + n * i_y], y - x2[i_x + 1 + n * i_y]
            res += beta(x_1, y_1, "bi-lineaire-faux", n) * img_ref[i_x + 1 + n * i_y]

        if i_y < n - 1:
            x_2, y_2 = x - x1[i_x + n * (i_y + 1)], y - x2[i_x + n * (i_y + 1)]
            res += beta(x_2, y_2, "bi-lineaire-faux", n) * img_ref[i_x + n * (i_y + 1)]

        if i_y > 0:
            x_3, y_3 = x - x1[i_x + n * (i_y - 1)], y - x2[i_x + n * (i_y - 1)]
            res += beta(x_3, y_3, "bi-lineaire-faux", n) * img_ref[i_x + n * (i_y - 1)]

        if i_x > 0:
            x_4, y_4 = x - x1[i_x - 1 + n * i_y], y - x2[i_x - 1 + n * i_y]
            res += beta(x_4, y_4, "bi-lineaire-faux", n) * img_ref[i_x - 1 + n * i_y]

        return res



def interp_from_neighbours(j, f_ref, n, x11, x22, x1, x2, interp, next_cell):
    l = j//n
    k = j%n
    if (l == 0):
        if (k == 0):
            ind_neighbours = [j, j+1, j+n+1, j+n]
        elif (k == n-1):
            ind_neighbours = [j-1, j, j+n, j+n+1]
        else:
            ind_neighbours = [j-1, j, j+1, j+n+1, j+n, j+n-1]
    elif (l == n-1):
        if (k == 0):
            ind_neighbours = [j-n, j-n+1, j+1, j]
        elif (k == n-1):
            ind_neighbours = [j-n-1, j-n, j, j-1]
        else:
            ind_neighbours = [j-n-1, j-n, j-n+1, j+1, j, j-1]
    elif (k == 0):
        ind_neighbours = [j-n, j-n+1, j+1, j+n+1, j+n, j]
    elif (k == n-1):
        ind_neighbours = [j-n-1, j-n, j, j+n, j+n-1, j-1]
    else:
        ind_neighbours = [j, j+1, j-1, j+n, j-n, j+n+1, j-n-1, j+n-1, j-n+1]

    res = 0
    for i in ind_neighbours:
        res += f_ref[i] * beta(x11[j] - x1[i], x22[j] - x2[i], interp, n)
        if beta(x11[j] - x1[i], x22[j] - x2[i], interp, n) != 0:
            next_cell.append(i)

    faux = 0
    if res == 0:
        # print("hypothèse voisins fausse", ind_neighbours)
        # exit(1)
        faux += 1

    return res, faux


def find_pixel_value(f_ref, i, j):
    n = int(math.sqrt(f_ref.size))
    if 0 <= i < n and 0 <= j < n:
        return f_ref[i + n * j]
    else:
        return 0 # pad with 0

def interp_img(f_ref, x11, x22, x1, x2, forme_interp):
    if forme_interp == "bilinear":
        return bilinear_interp(f_ref, x11, x22)

    elif forme_interp == "doors":
        return doors_interp(f_ref, x11, x22)

    else:
        N = x1.size
        f_t = np.zeros(f_ref.shape)
        n = int(math.sqrt(N))

        for i in range(f_t.size):
            i_x = int(x11[i] // (1./n))
            i_y = int(x22[i] // (1./n))

            if 0 <= i_x < n and n > i_y >= 0:
                f_t[i] = interp(x11[i], x22[i], x1, x2, forme_interp, n, (i_x, i_y), f_ref)

                # verification : interp et coefs interp donnent la même chose
                # coefs = coefs_interp(x11[j], x22[j], x1, x2, forme_interp, n, (j_x, j_y))
                # f_t[j] = coefs[1] * f_ref[j_x + n * j_y]
                # if j_x > 0:
                #     f_t[j] += coefs[0] * f_ref[j_x - 1 + n * j_y]
                # if j_x < n - 1:
                #     f_t[j] += coefs[2] * f_ref[j_x + 1 + n * j_y]
                # if j_y > 0:
                #     f_t[j] += coefs[3] * f_ref[j_x + n * (j_y - 1)]
                # if j_y < n - 1:
                #     f_t[j] += coefs[4] * f_ref[j_x + n * (j_y + 1)]
                # if f_t[j] - interp(x11[j], x22[j], x1, x2, forme_interp, n, (j_x, j_y), f_ref) > 1e-9:
                #     print("interp pas bonne : ", f_t[j], " != ", interp(x11[j], x22[j], x1, x2, forme_interp, n, (j_x, j_y), f_ref))
                    # exit(1)

        return f_t


def doors_interp(f_ref, x11, x22):
    N = f_ref.size
    f_t = np.zeros(f_ref.shape)
    n = int(math.sqrt(N))

    for i in range(N):
        y_low = math.floor(x22[i] * n)
        x_low = math.floor(x11[i] * n)
        y_high = math.ceil(x22[i] * n)
        x_high = math.ceil(x11[i] * n)

        dx = n * x11[i] - x_low
        dy = n * x22[i] - y_low

        if 0 <= dx < 0.5 and 0 <= dy < 0.5:
            f_t[i] = find_pixel_value(f_ref, x_low, y_low)
        elif 0 <= dx < 0.5 and 0.5 <= dy < 1:
            f_t[i] = find_pixel_value(f_ref, x_low, y_high)
        elif 0.5 <= dx < 1 and 0 <= dy < 0.5:
            f_t[i] = find_pixel_value(f_ref, x_high, y_low)
        elif 0.5 <= dx < 1 and 0.5 <= dy < 1:
            f_t[i] = find_pixel_value(f_ref, x_high, y_high)

    return f_t


def build_U_doors(x11, x22):
    N = x11.size
    n = int(math.sqrt(N))
    U = np.zeros((N, N))
    for i in range(N):
        y_low = math.floor(x22[i] * n)
        x_low = math.floor(x11[i] * n)
        y_high = math.ceil(x22[i] * n)
        x_high = math.ceil(x11[i] * n)

        dx = n * x11[i] - x_low
        dy = n * x22[i] - y_low

        if 0 <= x_low < n and 0 <= y_low < n:
            if 0 <= dx < 0.5 and 0 <= dy < 0.5:
                U[i, x_low + n * y_low] = 1

        if 0 <= x_low < n and 0 <= y_high < n:
            if 0 <= dx < 0.5 and 0.5 <= dy < 1:
                U[i, x_low + n * y_high] = 1

        if 0 <= x_high < n and 0 <= y_low < n:
            if 0.5 <= dx < 1 and 0 <= dy < 0.5:
                U[i, x_high + n * y_low] = 1

        if 0 <= x_high < n and 0 <= y_high < n:
            if 0.5 <= dx < 1 and 0.5 <= dy < 1:
                U[i, x_high + n * y_high] = 1

    return U

def bilinear_interp(f_ref, x11, x22):
    N = f_ref.size
    f_t = np.zeros(f_ref.shape)
    n = int(math.sqrt(N))

    for i in range(N):
        y_low = math.floor(x22[i] * n)
        x_low = math.floor(x11[i] * n)
        y_high = math.ceil(x22[i] * n)
        x_high = math.ceil(x11[i] * n)

        dx = n * x11[i] - x_low
        dy = n * x22[i] - y_low

        coin_gauche_haut = find_pixel_value(f_ref, x_low, y_low)
        coin_droit_haut = find_pixel_value(f_ref, x_high, y_low)
        coin_gauche_bas = find_pixel_value(f_ref, x_low, y_high)
        coin_droit_bas = find_pixel_value(f_ref, x_high, y_high)

        bord_haut = (1 - dx) * coin_gauche_haut + dx * coin_droit_haut
        bord_bas = (1 - dx) * coin_gauche_bas + dx * coin_droit_bas
        f_t[i] = (1 - dy) * bord_haut + dy * bord_bas

    return f_t


def build_U_bilinear(x11, x22):
    N = x11.size
    n = int(math.sqrt(N))
    U = np.zeros((N, N))
    for i in range(N):
        y_low = math.floor(x22[i] * n)
        x_low = math.floor(x11[i] * n)
        y_high = math.ceil(x22[i] * n)
        x_high = math.ceil(x11[i] * n)

        dx = n * x11[i] - x_low
        dy = n * x22[i] - y_low

        if 0 <= x_low < n and 0 <= y_low < n:
            U[i, x_low + n * y_low] = (1 - dx) * (1 - dy)

        if 0 <= x_low < n and 0 <= y_high < n:
            U[i, x_low + n * y_high] = (1 - dx) * dy

        if 0 <= x_high < n and 0 <= y_low < n:
            U[i, x_high + n * y_low] = dx * (1 - dy)

        if 0 <= x_high < n and 0 <= y_high < n:
            U[i, x_high + n * y_high] = dx * dy

    return U


def build_U_interp(x11, x22, forme_interp):
    if forme_interp == "doors":
        return build_U_doors(x11, x22)

    elif forme_interp == "bilinear":
        return build_U_bilinear(x11, x22)

    else:
        print("Nom d'interpolation non valide sur interp U")
        exit(1)


def fast_prod_hk_U(hk_p, hk_n, x11, x22, forme_interp):
    N = x11.size
    n = int(math.sqrt(N))
    res = np.zeros(N)

    for i in range(N):
        y_low = math.floor(x22[i] * n)
        x_low = math.floor(x11[i] * n)
        y_high = math.ceil(x22[i] * n)
        x_high = math.ceil(x11[i] * n)

        dx = n * x11[i] - x_low
        dy = n * x22[i] - y_low

        if forme_interp == "doors":

            if 0 <= x_low < n and 0 <= y_low < n:
                if 0 <= dx < 0.5 and 0 <= dy < 0.5:
                    res[x_low + n * y_low] += hk_p[i] - hk_n[i]

            if 0 <= x_low < n and 0 <= y_high < n:
                if 0 <= dx < 0.5 and 0.5 <= dy < 1:
                    res[x_low + n * y_high] += hk_p[i] - hk_n[i]

            if 0 <= x_high < n and 0 <= y_low < n:
                if 0.5 <= dx < 1 and 0 <= dy < 0.5:
                    res[x_high + n * y_low] += hk_p[i] - hk_n[i]

            if 0 <= x_high < n and 0 <= y_high < n:
                if 0.5 <= dx < 1 and 0.5 <= dy < 1:
                    res[x_high + n * y_high] += hk_p[i] - hk_n[i]


        elif forme_interp == "bilinear":
            if 0 <= x_low < n and 0 <= y_low < n:
                res[x_low + n * y_low] += (hk_p[i] - hk_n[i]) * (1 - dx) * (1 - dy)

            if 0 <= x_low < n and 0 <= y_high < n:
                res[x_low + n * y_high] += (hk_p[i] - hk_n[i]) * (1 - dx) * dy

            if 0 <= x_high < n and 0 <= y_low < n:
                res[x_high + n * y_low] += (hk_p[i] - hk_n[i]) * dx * (1 - dy)

            if 0 <= x_high < n and 0 <= y_high < n:
                res[x_high + n * y_high] += (hk_p[i] - hk_n[i]) * dx * dy

        else:
            print("No fast product implemented for this interpolation")
            exit(1)

    return res


def pad(cond1, cond2, cond3, cond4, val, cval=0):
    # pad array val with cval
    n = int(math.sqrt(val.size))
    val[cond1] = cval
    val[cond2] = cval
    val[cond3] = cval
    val[cond4] = cval

    return val



def vfast_prod_hk_U(hk, x11, x22, forme_interp):
    # very fast implementation working on arrays directly (hence with C), but I think some operations are useless
    N = x11.size
    n = int(math.sqrt(N))
    res = np.zeros(N)

    y_low = np.floor(x22 * n).astype(int)
    x_low = np.floor(x11 * n).astype(int)
    y_high = np.ceil(x22 * n).astype(int)
    x_high = np.ceil(x11 * n).astype(int)

    dx = n * x11 - x_low
    dy = n * x22 - y_low

    if forme_interp == "doors":
        val = hk

        val_low_low = val.copy()
        val_low_low = pad(x_low < 0, x_low >= n, y_low < 0, y_low >= n, val_low_low)
        val_low_low = pad(dx < 0, dx >= 0.5, dy < 0, dy >= 0.5, val_low_low)
        ind_low_low = pad(x_low < 0, x_low >= n, y_low < 0, y_low >= n,
                          x_low + n * y_low)  # just in order not to have out of range errors, but no problem with padding to 0 because values are also 0
        np.add.at(res, ind_low_low, val_low_low)

        val_low_high = val.copy()
        val_low_high = pad(x_low < 0, x_low >= n, y_high < 0, y_high >= n, val_low_high)
        val_low_high = pad(dx < 0, dx >= 0.5, dy < 0.5, dy >= 1, val_low_high)
        ind_low_high = pad(x_low < 0, x_low >= n, y_high < 0, y_high >= n, x_low + n * y_high)
        np.add.at(res, ind_low_high, val_low_high)

        val_high_low = val.copy()
        val_high_low = pad(x_high < 0, x_high >= n, y_low < 0, y_low >= n, val_high_low)
        val_high_low = pad(dx < 0.5, dx >= 1, dy < 0, dy >= 0.5, val_high_low)
        ind_high_low = pad(x_high < 0, x_high >= n, y_low < 0, y_low >= n, x_high + n * y_low)
        np.add.at(res, ind_high_low, val_high_low)

        val_high_high = val.copy()
        val_high_high = pad(x_high < 0, x_high >= n, y_high < 0, y_high >= n, val_high_high)
        val_high_high = pad(dx < 0.5, dx >= 1, dy < 0.5, dy >= 1, val_high_high)
        ind_high_high = pad(x_high < 0, x_high >= n, y_high < 0, y_high >= n, x_high + n * y_high)
        np.add.at(res, ind_high_high, val_high_high)

        # print("Pas de vfast pour doors mais c'est possible")
        # exit(1)

    elif forme_interp == "bilinear":
        val_low_low = hk * (1 - dx) * (1 - dy)
        val_low_low = pad(x_low < 0, x_low >= n, y_low < 0, y_low >= n, val_low_low)
        ind_low_low = pad(x_low < 0, x_low >= n, y_low < 0, y_low >= n, x_low + n * y_low) #just in order not to have out of range errors, but no problem with padding to 0 because values are also 0
        np.add.at(res, ind_low_low, val_low_low)

        val_low_high = hk * (1 - dx) * dy
        val_low_high = pad(x_low < 0, x_low >= n, y_high < 0, y_high >= n, val_low_high)
        ind_low_high = pad(x_low < 0, x_low >= n, y_high < 0, y_high >= n, x_low + n * y_high)
        np.add.at(res, ind_low_high, val_low_high)

        val_high_low = hk * dx * (1 - dy)
        val_high_low = pad(x_high < 0, x_high >= n, y_low < 0, y_low >= n, val_high_low)
        ind_high_low = pad(x_high < 0, x_high >= n, y_low < 0, y_low >= n, x_high + n * y_low)
        np.add.at(res, ind_high_low, val_high_low)

        val_high_high = hk * dx * dy
        val_high_high = pad(x_high < 0, x_high >= n, y_high < 0, y_high >= n, val_high_high)
        ind_high_high = pad(x_high < 0, x_high >= n, y_high < 0, y_high >= n, x_high + n * y_high)
        np.add.at(res, ind_high_high, val_high_high)

    else:
        print("No vfast implementation available")
        exit(1)

    return res
