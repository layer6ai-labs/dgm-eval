from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor

def pil_resize(x, output_size):
    s1, s2 = output_size
    def resize_single_channel(x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
    x = np.array(x.convert('RGB')).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x)/255