import inspect

from .encoder import Encoder
from .inception import InceptionEncoder
from .swav import ResNet50Encoder #ResNet18Encoder, ResNet18MocoEncoder, 
from .mae import VisionTransformerEncoder
from .data2vec import HuggingFaceTransformerEncoder
from .clip import CLIPEncoder
from .pixel import PixelEncoder
from .convnext import ConvNeXTEncoder
from .clip import CLIPEncoder
from .dinov2 import DINOv2Encoder
from .simclr import SimCLRResNetEncoder
MODELS = {
    "inception" : InceptionEncoder,
    "sinception" : InceptionEncoder,
    "mae": VisionTransformerEncoder,
    "data2vec": HuggingFaceTransformerEncoder,
    "swav": ResNet50Encoder,
    "clip": CLIPEncoder,
    'pixel': PixelEncoder,
    "convnext": ConvNeXTEncoder,
    "dinov2": DINOv2Encoder,
    "simclr": SimCLRResNetEncoder,
}


def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""

    model_cls = MODELS[model_name]

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:] # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    assert isinstance(encoder, Encoder), "Can only get representations with Encoder subclasses!"

    return encoder.to(device)
