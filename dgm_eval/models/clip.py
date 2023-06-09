import torch
import open_clip
from torchvision.transforms import Normalize, InterpolationMode
import torchvision.transforms.functional as TF
from torch.utils.checkpoint import checkpoint

from .encoder import Encoder
from ..resizer import pil_resize

ARCH_WEIGHT_DEFAULTS = {
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'datacomp_xl_s13b_b90k',
    'ViT-bigG-14': 'laion2b_s39b_b160k',
}
class CLIPEncoder(Encoder):
    def setup(self, arch:bool=None, pretrained_weights:bool=None, clean_resize:bool=False, depth:int=0):

        if arch is None:
            arch = 'ViT-L-14'
        if pretrained_weights is None:
            pretrained_weights=ARCH_WEIGHT_DEFAULTS[arch]

        self.model = open_clip.create_model(arch, pretrained_weights)
        self.clean_resize = clean_resize
        self.depth = depth

    def transform(self, image):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        size = self.model.visual.image_size
        if self.clean_resize:
            image = pil_resize(image, size)
        else:
            image = TF.resize(image, size, interpolation=InterpolationMode.BICUBIC).convert('RGB')
            image = TF.to_tensor(image)
        image  = TF.center_crop(image, size)
        return Normalize(mean, std)(image)

    def forward(self, x: torch.Tensor):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) +
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        blocks = self.model.visual.transformer.resblocks
        if self.depth<0:
            blocks = self.model.visual.transformer.resblocks[:self.depth]
        for r in blocks:
            x = r(x, attn_mask=None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.model.visual.global_average_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.model.visual.ln_post(x)

        if self.model.visual.proj is not None and self.depth==1:
            x = x @ self.model.visual.proj

        return x
