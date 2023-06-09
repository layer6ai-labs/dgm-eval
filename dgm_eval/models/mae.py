# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torchvision.transforms as TF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer

from .encoder import Encoder
from ..resizer import pil_resize
from .util.pos_embed import interpolate_pos_embed
import sys

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

MAE_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth'

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class VisionTransformerEncoder(Encoder):
    def setup(self, model=None, ckpt=None, clean_resize=False):


        # Model at https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
        checkpoint = load_state_dict_from_url(MAE_WEIGHTS_URL, progress=True)
        self.model = vit_large_patch16()
        self.clean_resize = clean_resize

        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint", file=sys.stderr)
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(self.model, checkpoint_model)

        # load pre-trained model
        msg = self.model.load_state_dict(checkpoint_model, strict=False)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        self.model.forward = self.model.forward_features

    def transform(self, image):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.resize(image, 224, interpolation=InterpolationMode.BICUBIC).convert('RGB')
            image = TF.to_tensor(image)
        # image  = TF.center_crop(image, size) TODO: Add crop center if it makes sense
        return TF.Normalize(imagenet_mean, imagenet_std)(image)
