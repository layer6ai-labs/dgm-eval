import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import sys

import numpy as np
import pandas as pd
import torch
from dgm_eval.dataloaders import get_dataloader
from dgm_eval.metrics import *
from dgm_eval.models import load_encoder, InceptionEncoder, MODELS
from dgm_eval.representations import get_representations, load_reps_from_path, save_outputs
from dgm_eval.heatmaps import visualize_heatmaps

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='dinov2', choices=MODELS.keys(),
                    help='Model to use for generating feature representations.')

parser.add_argument('--train_dataset', type=str, default='imagenet',
                    help='Dataset that model was trained on. Sets proper normalization for MAE.')

parser.add_argument('-bs', '--batch_size', type=int, default=50,
                    help='Batch size to use')

parser.add_argument('--num-workers', type=int, default=8,
                    help='Number of processes to use for data loading. '
                         'Defaults to `min(8, num_cpus)`')

parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

parser.add_argument('--nearest_k', type=int, default=5,
                    help='Number of neighbours for precision, recall, density, and coverage')

parser.add_argument('--reduced_n', type=int, default=10000,
                    help='Number of samples used for train, baseline, test, and generated sets for FLS')

parser.add_argument('--nsample', type=int, default=50000,
                    help='Maximum number of images to use for calculation')

parser.add_argument('--path', type=str, nargs='+',
                    help='Paths to the images, the first one is the real dataset, followed by generated')

parser.add_argument('--test_path', type=str, default=None,
                    help=('Path to test images'))

parser.add_argument('--metrics', type=str, nargs='+', default=['fd', 'fd-infinity', 'kd', 'prdc',
                                                               'is', 'authpct', 'ct', 'ct_test', 'ct_modified', 
                                                               'fls', 'fls_overfit', 'vendi', 'sw_approx'],
                    help="metrics to compute")

parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                    help='Path of model checkpoint.')

parser.add_argument('--arch', type=str, default=None,
                    help='Model architecture. If none specified use default specified in model class')

parser.add_argument('--heatmaps', action='store_true',
                    help='Generate heatmaps showing the fd focus on images.')

parser.add_argument('--heatmaps-perturbation', action='store_true',
                    help='Add some perturbation to the images on which gradcam is applied.')

parser.add_argument('--splits', type=int, default=10, help="num of splits for Inception Score(is)")

parser.add_argument('--output_dir', type=str, default='./experiment/default-test',
                    help='Directory to save outputs in')
parser.add_argument('--load_dir', type=str, default='./experiment/default-test',
                    help='Directory to save outputs in')
parser.add_argument('--exp_name', type=str, default='exp_00',
                    help='Directory to save outputs in')

parser.add_argument('--save', action='store_true',
                    help='Save representations to output_dir', default=True)

parser.add_argument('--load', action='store_true',
                    help='Load representations and statistics from previous runs if possible', default=True)

parser.add_argument('--no-load', action='store_false', dest='load',
                    help='Do not load representations and statistics from previous runs.')
parser.set_defaults(load=True)

parser.add_argument('--seed', type=int, default=13579,
                    help='Random seed')

parser.add_argument('--clean_resize', action='store_true',
                    help='Use clean resizing (from pillow)')

parser.add_argument('--depth', type=int, default=0,
                    help='Negative depth for internal layers, positive 1 for after projection head.')


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers
    print(device, num_workers)
    return device, num_workers


def get_dataloader_from_path(path, model_transform, num_workers, args, sample_w_replacement=False):
    print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)

    dataloader = get_dataloader(path, args.nsample, args.batch_size, num_workers, seed=args.seed,
                                sample_w_replacement=sample_w_replacement,
                                transform=lambda x: model_transform(x))

    return dataloader


def compute_representations(DL, model, device, args, dir_path=None):
    """
    Load representations from disk if path exists,
    else compute image representations using the specified encoder

    Returns:
        repsi: float32 (Nimage, ndim)
    """
    output_dir = args.output_dir if dir_path is None else dir_path

    if args.load:
        print(f'Loading saved representations from: {output_dir}\n', file=sys.stderr)
        repsi = load_reps_from_path(output_dir, args.model, None, DL)
        if repsi is not None: 
            return repsi
        print(f'No saved representations found: {output_dir}\n', file=sys.stderr)

    print('Calculating Representations\n', file=sys.stderr)
    repsi = get_representations(model, DL, device, normalized=False)
    if args.save:
        print(f'Saving representations to {args.output_dir}\n', file=sys.stderr)
        save_outputs(args.output_dir, repsi, args.model, None, DL)
    return repsi


def compute_scores(args, reps, test_reps, labels=None):

    scores={}
    vendi_scores = None

    if 'fd' in args.metrics:
        print("Computing FD \n", file=sys.stderr)
        scores['fd'] = compute_FD_with_reps(*reps)

    if 'fd_eff' in args.metrics:
        print("Computing Efficient FD \n", file=sys.stderr)
        scores['fd_eff'] = compute_efficient_FD_with_reps(*reps)

    if 'fd-infinity' in args.metrics:
        print("Computing fd-infinity \n", file=sys.stderr)
        scores['fd_infinity_value'] = compute_FD_infinity(*reps)

    if 'kd' in args.metrics:
        print("Computing KD \n", file=sys.stderr)
        mmd_values = compute_mmd(*reps)
        scores['kd_value'] = mmd_values.mean()
        scores['kd_variance'] = mmd_values.std()

    if 'prdc' in args.metrics:
        print("Computing precision, recall, density, and coverage \n", file=sys.stderr)
        reduced_n = min(args.reduced_n, reps[0].shape[0], reps[1].shape[0])
        inds0 = np.random.choice(reps[0].shape[0], reduced_n, replace=False)

        inds1 = np.arange(reps[1].shape[0])
        if 'realism' not in args.metrics:
            # Realism is returned for each sample, so do not shuffle if this metric is desired.
            # Else filenames and realism scores will not align
            inds1 = np.random.choice(inds1, min(inds1.shape[0], reduced_n), replace=False)

        prdc_dict = compute_prdc(
            reps[0][inds0], 
            reps[1][inds1], 
            nearest_k=args.nearest_k,
            realism=True if 'realism' in args.metrics else False)
        scores = dict(scores, **prdc_dict)

    try:
        if 'vendi' in args.metrics:
            print("Calculating diversity score", file=sys.stderr)
            scores['vendi'] = compute_vendi_score(reps[1])
            vendi_scores = compute_per_class_vendi_scores(reps[1], labels)
            scores['mean vendi per class'] = vendi_scores.mean()
    except:
        pass
    
    
    if 'authpct' in args.metrics:
        print("Computing authpct \n", file=sys.stderr)
        scores['authpct'] = compute_authpct(*reps)

    if 'sw_approx' in args.metrics:
        print('Aprroximating Sliced W2.', file=sys.stderr)
        scores['sw_approx'] = sw_approx(*reps)

    if 'ct' in args.metrics:
        print("Computing ct score \n", file=sys.stderr)
        scores['ct'] = compute_CTscore(reps[0], test_reps, reps[1])

    if 'ct_test' in args.metrics:
        print("Computing ct score, modified to identify mode collapse only \n", file=sys.stderr)
        scores['ct_test'] = compute_CTscore_mode(reps[0], test_reps, reps[1])

    if 'ct_modified' in args.metrics:
        print("Computing ct score, modified to identify memorization only \n", file=sys.stderr)
        scores['ct_modified'] = compute_CTscore_mem(reps[0], test_reps, reps[1])

    if 'fls' in args.metrics or 'fls_overfit' in args.metrics:
        train_reps, gen_reps = reps[0], reps[1]
        reduced_n = min(args.reduced_n, train_reps.shape[0]//2, test_reps.shape[0], gen_reps.shape[0])

        test_reps = test_reps[np.random.choice(test_reps.shape[0], reduced_n, replace=False)]
        gen_reps = gen_reps[np.random.choice(gen_reps.shape[0], reduced_n, replace=False)]

        print("Computing fls \n", file=sys.stderr)
        # fls must be after ot, as it changes train_reps
        train_reps = train_reps[np.random.choice(train_reps.shape[0], 2*reduced_n, replace=False)]
        train_reps, baseline_reps = train_reps[:reduced_n], train_reps[reduced_n:]

        if 'fls' in args.metrics:
            scores['fls'] = compute_fls(train_reps, baseline_reps, test_reps, gen_reps)
        if 'fls_overfit' in args.metrics:
            scores['fls_overfit'] = compute_fls_overfit(train_reps, baseline_reps, test_reps, gen_reps)

    for key, value in scores.items():
        if key=='realism': continue
        print(f'{key}: {value:.5f}\n')

    return scores, vendi_scores


def save_score(scores, output_dir, model, path, ckpt, nsample, is_only=False):

    ckpt_str = ''
    if ckpt is not None:
        ckpt_str = f'_ckpt-{os.path.splitext(os.path.basename(ckpt))[0]}'

    if is_only:
        out_str = f"Inception_score_{'-'.join([os.path.basename(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = f"fd_{model}_{'-'.join([os.path.basename(p) for p in path])}{ckpt_str}_nimage-{nsample}.txt"

    out_path = os.path.join(output_dir, out_str)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        for key, value in scores.items():
            if key=='realism': continue
            f.write(f"{key}: {value} \n")


def save_scores(scores, args, is_only=False, vendi_scores={}):

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_params = vars(args)
    run_params['reference_dataset'] = run_params['path'][0]
    run_params['test_datasets'] = run_params['path'][1:]

    ckpt_str = ''
    print(scores, file=sys.stderr)

    if is_only: 
        out_str = f'Inception_scores_nimage-{args.nsample}'
    else:
        out_str = f"{args.model}{ckpt_str}_scores_nimage-{args.nsample}"
    out_path = os.path.join(args.output_dir, out_str)

    np.savez(f'{out_path}.npz', scores=scores, run_params=run_params)

    if vendi_scores is not None and len(vendi_scores)>0:
        df = pd.DataFrame.from_dict(data=vendi_scores)
        out_str = f"{args.model}{ckpt_str}_vendi_scores_nimage-{args.nsample}"
        out_path = os.path.join(args.output_dir, out_str)
        print(f'saving vendi score to {out_path}.csv')
        df.to_csv(f'{out_path}.csv')

def get_inception_scores(args, device, num_workers):
    # The inceptionV3 with logit output is only used for calculate inception score
    print(f'Computing Inception score with model = inception, ckpt=None, and dims=1008.', file=sys.stderr)
    print('Loading Model', file=sys.stderr)

    IS_scores = {}
    model_IS = load_encoder('inception', device, ckpt=None,
                        dims=1008, arch=None,
                        pretrained_weights=None,
                        train_dataset=None,
                        clean_resize=args.clean_resize,
                        depth=args.depth)
    
    for i, path in enumerate(args.path[1:]):
        print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)
        dataloaderi = get_dataloader_from_path(args.path[i], model_IS.transform, num_workers, args)
        print(f'Computing inception score for {path}\n', file=sys.stderr)
        IS_score_i = compute_inception_score(model_IS, DataLoader=dataloaderi, splits=args.splits, device=device)
        IS_scores[f'{args.model}_{i:02d}'] = IS_score_i
        print(IS_score_i)
    save_scores(IS_scores, args, is_only=True)
    if len(args.metrics) == 1: sys.exit(0)

    return IS_scores

def main():
    args = parser.parse_args()

    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)
    print(args)

    IS_scores = None
    if 'is' in args.metrics and args.model == 'inception':
        # Does not require a reference dataset, so compute first.
        IS_scores = get_inception_scores(args, device, num_workers)
       
    print('Loading Model', file=sys.stderr)
    # Get train representations
    model = load_encoder(args.model, device, ckpt=None, arch=None,
                        clean_resize=args.clean_resize,
                        sinception=True if args.model=='sinception' else False,
                        depth=args.depth,
                        )
    dataloader_real = get_dataloader_from_path(args.path[0], model.transform, num_workers, args)
    
    print("Real train data representations")
    reps_real = compute_representations(dataloader_real, model, device, args, dir_path=args.load_dir)

    # Get test representations
    repsi_test = None
    if args.test_path is not None:
        print("Real test data representations")
        dataloader_test = get_dataloader_from_path(args.test_path, model.transform, num_workers, args)

        repsi_test = compute_representations(dataloader_test, model, device, args, dir_path=args.load_dir)

    # Loop over all generated paths
    all_scores = {}
    vendi_scores = {}
    for i, path in enumerate(args.path[1:]):
        print("Generated data representations")
        # Saves baseline representations for multiple evals
        if len(args.path) == 3 and i==1:
            outdir = args.load_dir
            exp_name = "baseline"
        else:
            outdir = args.output_dir
            exp_name = args.exp_name
        
        print(i, exp_name, outdir,"\n\n\n")
            
        dataloaderi = get_dataloader_from_path(path, model.transform, num_workers, args,
                                               sample_w_replacement=True if ':train' in path else False)
        
        repsi = compute_representations(dataloaderi, model, device, args, dir_path=outdir)
        reps = [reps_real, repsi]

        print(f'Computing scores between reference dataset and {path}\n', file=sys.stderr)
        scores_i, vendi_scores_i = compute_scores(args, reps, repsi_test, dataloaderi.labels)
        if vendi_scores_i is not None:
            vendi_scores[os.path.basename(path)] = vendi_scores_i

        print('Saving scores\n', file=sys.stderr)
        save_score(
            scores_i, args.output_dir, args.model, [args.path[0], path], None, args.nsample,
        )
        if IS_scores is not None:
            scores_i.update(IS_scores[f'{exp_name}_{i:02d}'])
        all_scores[f'{exp_name}_{i:02d}'] = scores_i

        if args.heatmaps:
            print('Visualizing FD gradient with gradcam\n', file=sys.stderr)
            heatmap_suffix = f"{args.model}_{dataloader_real.dataset_name}_{dataloaderi.dataset_name}" + \
                             f"{'_perturbation' if args.heatmaps_perturbation else ''}_{args.seed}"
            visualize_heatmaps(reps_real, repsi, model, dataset=dataloaderi.data_set, results_dir=args.output_dir,
                               results_suffix=heatmap_suffix, dataset_name=dataloaderi.dataset_name, device=device,
                               perturbation=args.heatmaps_perturbation, random_seed=args.seed)
    # save scores from all generated paths
    save_scores(all_scores, args, vendi_scores=vendi_scores)
    df = pd.DataFrame.from_dict(data=all_scores)
   
    if len(args.path) == 3:
        # calculate difference betwwen the first and second column on pandas
        df['Gain'] = df.iloc[:, 0] - df.iloc[:, 1]
    
    df.round(2).to_csv(os.path.join(args.output_dir, 'scores.csv'))
if __name__ == '__main__':
    main()
