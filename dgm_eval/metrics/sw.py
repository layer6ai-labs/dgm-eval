from numpy import linalg

def sw_approx(X, Y):
    '''Approximate Sliced W2 without
    Monte Carlo From https://arxiv.org/pdf/2106.15427.pdf'''
    d = X.shape[1]
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    mean_term = linalg.norm(mean_X - mean_Y) ** 2 / d
    m2_Xc = (linalg.norm(X - mean_X, axis=1) ** 2).mean() / d
    m2_Yc = (linalg.norm(Y - mean_Y, axis=1) ** 2).mean() / d
    approx_sw = (mean_term + (m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2)) ** 2) ** (1/2)
    return approx_sw
