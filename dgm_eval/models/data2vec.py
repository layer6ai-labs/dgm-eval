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

from transformers import Data2VecVisionConfig, Data2VecVisionModel
from transformers import AutoFeatureExtractor, AutoModel, AutoConfig, AutoImageProcessor

from .encoder import Encoder

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# DATA2VEC_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt'

from transformers import Data2VecVisionConfig, Data2VecVisionModel


class HuggingFaceTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(HuggingFaceTransformer, self).__init__(**kwargs)

        # checkpoint = load_state_dict_from_url(DATA2VEC_WEIGHTS_URL, progress=True)
        # print(checkpoint)    

        # self.model = AutoModel.from_pretrained("facebook/data2vec-vision-base", add_pooling_layer=True)
        self.model = AutoModel.from_pretrained("facebook/data2vec-vision-large", add_pooling_layer=True)

    def forward(self, inputs, mask=None, **kwargs):

        outputs = self.model(inputs, return_dict=True)

        # print('Encoder out shape = ', encoder_out.shape)
        return outputs.pooler_output


class HuggingFaceTransformerEncoder(Encoder):
    def setup(self, ckpt=None):

        # self.image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-large")

        self.model = HuggingFaceTransformer()

    def transform(self, image):
        return self.image_processor(image, return_tensors="pt")
