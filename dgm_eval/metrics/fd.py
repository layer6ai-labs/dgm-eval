import numpy as np
from scipy import linalg
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def compute_statistics(reps):
    """Compute necessary statistics from representtions"""
    mu = np.mean(reps, axis=0)
    sigma = np.cov(reps, rowvar=False)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fd calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # Return mean and covariance terms and intermediate steps, as well as FD
    mean_term = diff.dot(diff)
    tr1, tr2 = np.trace(sigma1), np.trace(sigma2)
    cov_term = tr1 + tr2 - 2 * tr_covmean

    return mean_term+cov_term

def compute_FD_with_reps(reps1, reps2, eps=1e-6):
    """
    Params:
    -- reps1   : activations of a representative data set (usually train)
    -- reps2   : activations of generated data set
    Returns:
    --   : The Frechet Distance.
    """
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)
    return compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=eps)


def compute_efficient_FD_with_reps(reps1, reps2):
    """
    A more efficient computation of FD as proposed at the following link:
    https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/

    Confirmed to return identical values as the standard calculation above on all datasets we in our work.
    """
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)
    sqrt_trace = np.real(linalg.eigvals(sigma1 @ sigma2)**0.5).sum()
    return ((mu1 - mu2)**2).sum() + sigma1.trace() + sigma2.trace() - 2 * sqrt_trace



def compute_FD_infinity(reps1, reps2, num_points=15):
    '''
    reps1:
        representation of training images
    reps2:
        representatio of generated images
    num_points:
        Number of FD_N we evaluate to fit a line.
        Default: 15

    '''
    fds = []

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fd_batches = np.linspace(min(5000, max(len(reps2)//10, 2)), len(reps2), num_points).astype('int32')
    mu1, sigma1 = compute_statistics(reps1)

    pbar = tqdm(total=num_points, desc='FID-infinity batches')
    # Evaluate FD_N
    rng = np.random.default_rng()
    for fd_batch_size in fd_batches:
        # sample, replacement allowed for different sample sizes
        fd_activations = rng.choice(reps2, fd_batch_size, replace=False)
        mu2, sigma2 = compute_statistics(fd_activations)
        fds.append(compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=1e-6))
        pbar.update(1)
    del pbar
    fds = np.array(fds).reshape(-1, 1)

    # Fit linear regression
    reg = LinearRegression().fit(1/fd_batches.reshape(-1, 1), fds)
    fd_infinity = reg.predict(np.array([[0]]))[0,0]

    return fd_infinity
