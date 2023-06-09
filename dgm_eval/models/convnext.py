import open_clip
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
import torchvision.transforms.functional as TF

from .encoder import Encoder
from ..resizer import pil_resize
from timm.models import create_model
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import sys

class ConvNeXTEncoder(Encoder):
    """
    requires timm version: 0.8.19.dev0
    model_arch options: 
        convnext_xlarge_in22k (imagenet 21k); default
        convnext_xxlarge.clip_laion2b_rewind (clip objective trained on laion2b)

    see more options https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

    """
    def setup(self, arch='convnext_large_in22k', clean_resize=False):
        if arch==None: arch = 'convnext_large_in22k'
        self.arch = arch
        self.model = create_model(
                    arch,
                    pretrained=True,
                )
        self.model.eval()

        if arch == "convnext_large_in22k":
            self.input_size = 224
        elif arch in ["convnext_base.clip_laion2b_augreg", "convnext_xxlarge.clip_laion2b_rewind"]:
            self.input_size = 256

        self.clean_resize = clean_resize
        self.build_transform()


    def build_transform(self):
        # get mean & std based on the model arch
        if self.arch == "convnext_large_in22k":
            print("IMAGENET MEAN STD", file=sys.stderr)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        elif "clip" in self.arch:
            print("OPENAI CLIP MEAN STD", file=sys.stderr)
            mean = OPENAI_CLIP_MEAN
            std = OPENAI_CLIP_STD

        t = []

        # warping (no cropping) when evaluated at 384 or larger
        if self.input_size >= 384:
            t.append(
            transforms.Resize((self.input_size, self.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {self.input_size} size input images...", file=sys.stderr)
        else:
            size = 256
            t.append(
                # to maintain same ratio
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(self.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        self.transform_ops = transforms.Compose(t)
        return

    def transform(self, image):
        return self.transform_ops(image)

    def forward(self, x):
        # forward features + global_pool + norm + flatten => output dims ()
        outputs = self.model.forward_features(x)
        outputs = self.model.head.global_pool(outputs)
        outputs = self.model.head.norm(outputs)
        outputs = self.model.head.flatten(outputs)
        return outputs
