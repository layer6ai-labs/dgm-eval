# a refactored version from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import torch
from tqdm import tqdm

import numpy as np
from scipy.stats import entropy

def compute_inception_score(model, DataLoader=None, splits=10, device=None):
    """Computes the inception score of the generated images imgs"""
    score = {}
    preds = get_preds(model, DataLoader, device)
    inecption_score, std = calculate_score(preds, splits=splits, N=DataLoader.nimages)
    score['inception score'] = inecption_score
    score['inception std'] = std
    return score

def get_preds(model, DataLoader, device):
    model.eval()
    start_idx = 0

    for ibatch, batch in enumerate(tqdm(DataLoader.data_loader)):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
            if not torch.is_tensor(pred): # Some encoders output tuples or lists
                pred = pred[0]
        pred = pred.cpu().numpy()


        if ibatch==0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            pred_arr = np.empty((DataLoader.nimages, dims))

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    return pred_arr

def calculate_score(preds, splits=10, N=50000, shuffle=True, rng_seed=2020):
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        preds = preds[rng.permutation(N), :]
    # Compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
