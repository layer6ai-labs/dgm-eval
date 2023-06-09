#https://github.com/marcojira/fls/blob/main/metrics/AuthPct.py
import torch

def compute_authpct(train_feat, gen_feat):
    with torch.no_grad():
        train_feat = torch.tensor(train_feat, dtype=torch.float32)
        gen_feat = torch.tensor(gen_feat, dtype=torch.float32)
        real_dists = torch.cdist(train_feat, train_feat)

        # Hacky way to get it to ignore distance to self in nearest neighbor calculation
        real_dists.fill_diagonal_(float("inf"))
        gen_dists = torch.cdist(train_feat, gen_feat)

        real_min_dists = real_dists.min(axis=0)
        gen_min_dists = gen_dists.min(dim=0)

        # For every synthetic point, find its closest real point, d1
        # Then, for that real point, find its closest real point(not itself), d2
        # if d2<d1, then its authentic
        authen = real_min_dists.values[gen_min_dists.indices] < gen_min_dists.values
        authpct = (100 * torch.sum(authen) / len(authen)).item()
    return authpct
