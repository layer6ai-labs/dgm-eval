import os
import json

import numpy as np
import torch
from tqdm import tqdm

from dgm_eval.heatmaps.gradcam import GradCAM
from dgm_eval.heatmaps.heatmaps_utils import get_image, create_grid, perturb_image, zero_one_scaling
from dgm_eval.models.encoder import Encoder


def visualize_heatmaps(reps_real: np.array,
                       reps_gen: np.array,
                       model: Encoder,
                       dataset: torch.utils.data.Dataset,
                       results_dir: str,
                       results_suffix: str = 'default',
                       dataset_name: str = None,
                       num_rows: int = 4,
                       num_cols: int = 4,
                       device: torch.device = torch.device('cpu'),
                       perturbation: bool = False,
                       human_exp_indices: str = None,
                       random_seed: int = 0) -> None:
    """Visualizes to which regions in the images FID is the most sensitive to."""

    visualizer = GradCAM(model, reps_real, reps_gen, device)

    # ----------------------------------------------------------------------------
    # Visualize FID sensitivity heatmaps.
    heatmaps, labels, images = [], [], []

    # Sampling image indices
    rnd = np.random.RandomState(random_seed)
    if human_exp_indices is not None:
        with open(human_exp_indices, 'r') as f_in:
            index_to_score = json.load(f_in)
        indices = [int(idx) for idx in list(index_to_score.keys()) if int(idx) < len(dataset)]
        if len(indices) < len(index_to_score):
            raise RuntimeWarning("The datasets were subsampled so the human experiment indices will not be accurate. "
                               "Please use '--nmax_images' with a higher value")
        vis_images_indices = [idx for idx in rnd.choice(indices, size=num_rows * num_cols, replace=False)]
        vis_images_scores = [index_to_score[str(idx)] for idx in vis_images_indices]
        vis_images_indices = [idx for _, idx in sorted(zip(vis_images_scores, vis_images_indices))] # sorting indices in ascending human score
    else:
        vis_images_indices = rnd.choice(np.arange(len(dataset)), size=num_rows * num_cols, replace=False)

    print('Visualizing heatmaps...')
    for idx in tqdm(vis_images_indices):

        # ----------------------------------------------------------------------------
        # Get selected image and do required transforms
        image = get_image(dataset, idx, device, perturbation=perturbation)

        # ----------------------------------------------------------------------------
        # Compute and visualize a sensitivity map.
        heatmap, label = visualizer.get_map(image, idx)

        heatmaps.append(heatmap)
        labels.append(label)
        images.append(np.clip(zero_one_scaling(image=image.detach().cpu().numpy().squeeze(0)) * 255, 0.0, 255.0).astype(np.uint8))

    human_scores = labels
    if human_exp_indices is not None:
        human_scores = [f"{index_to_score[str(idx)]:0.2f}" for idx in vis_images_indices]

    # ----------------------------------------------------------------------------
    # Create a grid of overlay heatmaps.
    heatmap_grid = create_grid(images=heatmaps, labels=labels, num_rows=num_rows, num_cols=num_cols)
    image_grid = create_grid(images=images, labels=human_scores, num_rows=num_rows, num_cols=num_cols)
    heatmap_grid.save(os.path.join(results_dir, f'sensitivity_grid_{results_suffix}.png'))
    image_grid.save(os.path.join(results_dir, f'images_grid_{results_suffix}.png'))
