from typing import Any, List

import numpy as np
import torch

from dgm_eval.heatmaps.heatmaps_utils import get_features, show_heatmap_on_image


class GradCAM:

    def __init__(self, model, reps_real, reps_gen, device, **kwargs):
        # Register forward and backward hooks to get activations and gradients, respectively.
        self.acts_and_gradients = ActivationsAndGradients(network=model, model_name=model.name)

        # Compute feature statistics on real images.
        self.mean_reals = torch.from_numpy(np.mean(reps_real, axis=0)).to(device)
        self.cov_reals = torch.from_numpy(np.cov(reps_real, rowvar=False)).to(device)

        self.reps_gen = reps_gen
        self.model = model
        self.device = device

    def get_map(self, image, idx):
        """
        Compute heatmap from the gradients and activation of the Frechet distance
        Return the heatmap and image label if possible
        """
        self.acts_and_gradients.eval()  # Model needs to be in eval mode

        # Computing selected image features
        features = get_features(self.acts_and_gradients, image)

        # Compute feature statistics without the selected image
        mean_gen = torch.from_numpy(np.mean(np.delete(self.reps_gen, idx, axis=0), axis=0)).to(self.device)
        cov_gen = torch.from_numpy(np.cov(np.delete(self.reps_gen, idx, axis=0), rowvar=False)).to(self.device)

        # Compute fid without the selected image
        original_fid = wasserstein2_loss(mean_reals=self.mean_reals, mean_gen=mean_gen,
                                         cov_reals=self.cov_reals, cov_gen=cov_gen)

        # Updating feature statistics with the selected image to get gradients
        num_images = len(self.reps_gen)
        mean = ((num_images - 1) / num_images) * mean_gen + (1 / num_images) * features
        cov = ((num_images - 2) / (num_images - 1)) * cov_gen + \
              (1 / num_images) * torch.mm((features - mean_gen).T, (features - mean_gen))

        # Compute frechet distance and back-propagate loss
        loss = wasserstein2_loss(mean_reals=self.mean_reals, mean_gen=mean,
                                 cov_reals=self.cov_reals, cov_gen=cov)
        loss.backward()
        delta_fid = loss.detach().cpu().numpy() - original_fid.detach().cpu().numpy()

        # Get heatmap from gradients and activation
        heatmap = self._get_heatmap_from_grads()

        # Get overlay of heatmap on image
        overlay = show_heatmap_on_image(heatmap, image)

        # Get classification label if possible
        label = None
        if hasattr(self.model, 'get_label'):
            label = self.model.get_label(features)
        # label = f"{delta_fid:0.5f}"

        return overlay, label

    def _get_heatmap_from_grads(self):
        # Get activations and gradients from the target layer by accessing hooks.
        activations = self.acts_and_gradients.activations[-1]
        gradients = self.acts_and_gradients.gradients[-1]

        if len(activations.shape) == 3:
            dim = int(activations.shape[-1] ** 0.5)
            activations = activations[:, :, 1:].reshape(*activations.shape[:-1], dim, dim)
            gradients = gradients[:, :, 1:].reshape(*gradients.shape[:-1], dim, dim)

        # Turn gradients and activation into heatmap.
        weights = np.mean(gradients ** 2, axis=(2, 3), keepdims=True)
        heatmap = (weights * activations).sum(axis=1)

        return heatmap[0]


MODEL_TO_LAYER_NAME_MAP = {
    'inception': 'blocks.3.2',
    'clip': 'visual.transformer.resblocks.11.ln_1',
    'mae': 'blocks.23.norm1',
    'swav': 'layer4.2',
    'dinov2': 'blocks.23.norm1',
    'convnext': 'stages.3.blocks.2',
    'data2vec': 'model.encoder.layer.23.layernorm_before',
    'simclr': 'net.4.blocks.2.net.3'
}

MODEL_TO_TRANSFORM_MAP = {
    'inception': lambda x : x,
    'clip': lambda x : -x.transpose(1, 2, 0),
    'mae': lambda x : x.transpose(0, 2, 1),
    'swav': lambda x : x,
    'dinov2': lambda x : -x.transpose(0, 2, 1),
    'convnext': lambda x: -x,
    'data2vec': lambda x: x.transpose(0, 2, 1),
    'simclr': lambda x: -x,
}


class ActivationsAndGradients:
    """Class to obtain intermediate activations and gradients.
       Adapted from: https://github.com/jacobgil/pytorch-grad-cam"""

    def __init__(self,
                 network: Any,
                 model_name: str,
                 network_kwargs: dict = None) -> None:
        self.network = network
        self.network_kwargs = network_kwargs if network_kwargs is not None else {}
        self.gradients: List[np.ndarray] = []
        self.activations: List[np.ndarray] = []
        self.transform = MODEL_TO_TRANSFORM_MAP.get(model_name)

        target_layer_name = MODEL_TO_LAYER_NAME_MAP.get(model_name)
        target_layer = dict(network.model.named_modules()).get(target_layer_name)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self,
                        module: Any,
                        input: Any,
                        output: Any) -> None:
        """Saves forward pass activations."""
        activation = output
        self.activations.append(self.transform(activation.detach().cpu().numpy()))

    def save_gradient(self,
                      module: Any,
                      grad_input: Any,
                      grad_output: Any) -> None:
        """Saves backward pass gradients."""
        # Gradients are computed in reverse order.
        grad = grad_output[0]
        self.gradients = [self.transform(grad.detach().cpu().numpy())] + self.gradients  # Prepend current gradients.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Resets hooked activations and gradients and calls model forward pass."""
        self.gradients = []
        self.activations = []
        return self.network(x, **self.network_kwargs)

    def eval(self):
        self.network.eval()


def wasserstein2_loss(mean_reals: torch.Tensor,
                      mean_gen: torch.Tensor,
                      cov_reals: torch.Tensor,
                      cov_gen: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """Computes 2-Wasserstein distance."""
    mean_term = torch.sum(torch.square(mean_reals - mean_gen.squeeze(0)))
    eigenvalues = torch.real(torch.linalg.eig(torch.matmul(cov_gen, cov_reals))[0])
    cov_term = torch.trace(cov_reals) + torch.trace(cov_gen) - 2 * torch.sum(torch.sqrt(abs(eigenvalues) + eps))
    return mean_term + cov_term
