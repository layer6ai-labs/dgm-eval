import shutil
import pytest
import os
import torch
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


SWAV_WEIGHTS_URL = 'https://www.dropbox.com/s/ghjrgktg0lyjorn/swav_800ep_pretrain.pth.tar?dl=1'
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501
MAE_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth'


def weight_loading_helper(url, device, file_name):
    test_dir = './test_download'
    os.mkdir(test_dir)
    load_state_dict_from_url(url, progress=True, model_dir=test_dir, file_name=file_name, map_location=device)
    assert os.path.exists(f'{test_dir}/{file_name}'), f'can\'t download {file_name} from url {url}'
    shutil.rmtree(test_dir, ignore_errors=False, onerror=None)


def test_load_fid_weights_cpu():
    weight_loading_helper(FID_WEIGHTS_URL, torch.device('cpu'), 'fid_weights')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu available')
def test_load_fid_weights_gpu():
    weight_loading_helper(FID_WEIGHTS_URL, torch.device('cuda'), 'fid_weights')


def test_load_swav_weights_cpu():
    weight_loading_helper(SWAV_WEIGHTS_URL, torch.device('cpu'), 'swav_weights')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu available')
def test_load_swav_weights_gpu():
    weight_loading_helper(SWAV_WEIGHTS_URL, torch.device('cuda'), 'swav_weights')


def test_load_mae_weights_cpu():
    weight_loading_helper(MAE_WEIGHTS_URL, torch.device('cpu'), 'mae_weights')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu available')
def test_load_mae_weights_gpu():
    weight_loading_helper(MAE_WEIGHTS_URL, torch.device('cuda'), 'mae_weights')
