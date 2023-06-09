import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torchvision.transforms.functional as TF


from .encoder import Encoder

from ..resizer import pil_resize

class PixelEncoder(Encoder):
    def setup(self):
        self.model = torch.nn.Identity()
        pass

    def transform(self, image):
        image = pil_resize(image, (32, 32))
        return image
