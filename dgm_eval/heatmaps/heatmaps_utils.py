from typing import List, Optional, Tuple

import PIL
import cv2
import numpy as np
import torch


def get_image(dataset, idx, device, perturbation=False):
    image = dataset[idx]
    if isinstance(image, tuple):
        # image is likely tuple[images, label]
        image = image[0]
    if isinstance(image, torch.Tensor):
        # add batch dimension
        image.unsqueeze_(0)
    else:  # Special case of data2vec
        image = image.data["pixel_values"]
    # Convert grayscale to RGB
    if image.ndim == 3:
        image.unsqueeze_(1)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if perturbation:
        image = perturb_image(image)
    image = image.to(device)
    image.requires_grad = True
    return image


def get_features(model, image):
    features = model(image)[0]

    if not torch.is_tensor(features):  # Some encoders output tuples or lists
        features = features[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if features.dim() > 2:
        if features.size(2) != 1 or features.size(3) != 1:
            features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))

        features = features.squeeze(3).squeeze(2)

    if features.dim() == 1:
        features = features.unsqueeze(0)

    return features


def zero_one_scaling(image: np.ndarray) -> np.ndarray:
    """Scales an image to range [0, 1]."""
    if np.all(image == 0):
        return image
    image = image.astype(np.float32)
    if (image.max() - image.min()) == 0:
        return image
    return (image - image.min()) / (image.max() - image.min())


def show_heatmap_on_image(heatmap, image, colormap: int = cv2.COLORMAP_PARULA, heatmap_weight: float = 1.):
    image_np = image.detach().cpu().numpy()[0]
    _, h, w = image_np.shape

    # Scale heatmap values between 0 and 255.
    heatmap = zero_one_scaling(image=heatmap)
    heatmap = np.clip((heatmap * 255.0).astype(np.uint8), 0.0, 255.0)

    # Scale to original image size.
    heatmap = np.array(
        PIL.Image.fromarray(heatmap).resize((w, h), resample=PIL.Image.LANCZOS).convert(
            'L'))

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image=image_np)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image
    overlay = zero_one_scaling(image=overlay)
    overlay = np.clip(overlay * 255, 0.0, 255.0).astype(np.uint8)

    return overlay


def create_grid(images: List[np.ndarray],
                num_rows: int,
                num_cols: int,
                labels: Optional[List[str]] = None,
                label_loc: Tuple[int, int] = (0, 0),
                fontsize: int = 32,
                font_path: str = './data/times-new-roman.ttf') -> PIL.Image:
    """Creates an image grid."""
    h, w = 256, 256
    if labels is None or len(labels)==0:
        labels = [None]*len(images)
    assert len(images) == len(labels)
    font = PIL.ImageFont.truetype(font_path, fontsize)

    grid = PIL.Image.new('RGB', size=(num_cols * h, num_rows * w))

    for i in range(num_rows):
        for j in range(num_cols):
            im = cv2.resize(images.pop(0).transpose((1, 2, 0)), dsize=(h, w), interpolation=cv2.INTER_CUBIC)
            im = PIL.Image.fromarray(im)

            label = labels.pop(0)
            if label is not None:
                draw = PIL.ImageDraw.Draw(im)
                draw.text(label_loc, f'{label}'.capitalize(), font=font)

            grid.paste(im, box=(j * w, i * h))
    return grid


def perturb_image(image):
    # image is (B, N, H, W)
    _, _, h, w = image.shape
    image[:, :, int(2*h/10):int(3*h/10), int(2*w/10):int(3*w/10)] = 0
    return image
