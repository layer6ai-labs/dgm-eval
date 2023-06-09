from .inception import InceptionEncoder
from .swav import ResNet50Encoder #, ResNet18Encoder
from .mae import VisionTransformerEncoder
from .data2vec import HuggingFaceTransformerEncoder
from .clip import CLIPEncoder
from .convnext import ConvNeXTEncoder
from .dinov2 import DINOv2Encoder
from .load_encoder import load_encoder, MODELS
from .simclr import SimCLRResNetEncoder