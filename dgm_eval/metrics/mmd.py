from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np

def compute_mmd(feat_real, feat_gen, n_subsets=100, subset_size=1000, **kernel_args):
    m = min(feat_real.shape[0], feat_gen.shape[0])
    subset_size = min(subset_size, m)
    mmds = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD') as bar:
        for i in bar:
            g = feat_real[choice(len(feat_real), subset_size, replace=False)]
            r = feat_gen[choice(len(feat_gen), subset_size, replace=False)]
            o = compute_polynomial_mmd(g, r, **kernel_args) 
            mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return mmds


def compute_polynomial_mmd(feat_r, feat_gen, degree=3, gamma=None, coef0=1):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = feat_r
    Y = feat_gen

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY)


def _mmd2_and_variance(K_XX, K_XY, K_YY):
    # based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
    mmd2 -= 2 * K_XY_sum / (m * m)
    return mmd2
