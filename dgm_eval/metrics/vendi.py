from sklearn import preprocessing
from sklearn.metrics.pairwise import polynomial_kernel
import scipy
import scipy.linalg
import numpy as np
from tqdm import tqdm

def compute_vendi_score(X, q=1, normalize=True, kernel='linear'):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    if kernel == 'linear':
        S = X @ X.T
    elif kernel == 'polynomial':
        S = polynomial_kernel(X, degree=3, gamma=None, coef0=1)  # currently hardcoding kernel params to match KID
    else:
        raise NotImplementedError("kernel not implemented")
    # print('similarity matrix of shape {}'.format(S.shape))
    w = scipy.linalg.eigvalsh(S / n)
    return np.exp(entropy_q(w, q=q))

def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)

def compute_per_class_vendi_scores(reps, labels):
    num_classes = len(np.unique(labels))
    vendi_per_class = np.zeros(shape=num_classes)
    with tqdm(total=num_classes) as pbar:
        for i in range(num_classes):
            reps_class = reps[labels==i]
            vendi_per_class[i] = compute_vendi_score(reps_class)
            pbar.update(1)
    return vendi_per_class
