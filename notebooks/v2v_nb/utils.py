import json
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import video2vision as v2v

from .config import get_config, ParamNotSet

__all__ = [
    'coefficient_of_determination', 'gamma_scale', 'get_cache_path',
    'get_loader', 'get_shift', 'load_csv', 'load_operator', 'make_displayable',
    'mean_absolute_error', 'signal_to_noise_ratio',
]


def coefficient_of_determination(gt: np.ndarray, preds: np.ndarray) \
        -> np.ndarray:
    '''
    Calculates coefficient of determination.
    '''
    ss_res, ss_tot = ((gt - preds)**2).sum(), ((gt - gt.mean())**2).sum()
    return 1 - (ss_res / ss_tot)


def gamma_scale(image):
    '''
    Performs gamma scaling on an image prior to display.
    '''
    # Coerce to [0, 1] range
    pix_min = image.min((0, 1), keepdims=True)
    pix_max = image.max((0, 1), keepdims=True)
    image = (image - pix_min) / np.clip(pix_max - pix_min, 1e-6, 1)
    return image ** 2.2


def get_cache_path(which: str) -> Optional[str]:
    config = get_config()

    if not config.use_cache:
        return None
    return os.path.join(config['experiment_name'], f'{which}.json')


def get_loader(which: str) -> v2v.Loader:
    config = get_config()

    if config[which] is None:
        raise ParamNotSet(which)

    return v2v.Loader(config[which], config.image_size)


def get_shift(which: str) -> int:
    config = get_config()

    if not which.startswith(('uv_', 'vis_')):
        raise KeyError(which)

    root_key = 'vis_path' if which.startswith('vis_') else 'uv_path'
    for k in ['shift', root_key, which]:
        if config[k] is None:
            raise ParamNotSet('shift')

    if config[root_key] != config[which]:
        return 0
    elif which.startswith('vis_'):
        return max(config['shift'], 0)
    else:
        return max(-config['shift'], 0)


def load_csv(path: str) -> np.ndarray:
    out = np.genfromtxt(path, skip_header=True, delimiter=',')
    out = out[:, 1:]
    if out.max() > 2:
        out /= 100
    return out


def load_operator(path: str) -> v2v.Operator:
    with open(path, 'r') as op_file:
        x = json.load(op_file)
    op_cls = v2v.OPERATOR_REGISTRY.get(x.pop('class'))
    return op_cls(**x)


def make_displayable(*images: np.ndarray) -> Image:
    '''
    Converts one or more images into a form that can be displayed with the
    display function in a notebook.
    '''
    hs, ws = zip(*(image.shape[:2] for image in images))
    h, w = max(hs), sum(ws)
    display_image = Image.new('RGB', (w, h))
    x = 0

    for image in images:
        image = gamma_scale(image)
        if image.dtype in {np.float32, np.float64}:
            image = np.clip(256 * image, 0, 255).astype(np.uint8)
        if (image.ndim == 3) and (image.shape[2] == 1):
            image = image.reshape(*image.shape[:2])
        fmt = 'L' if (image.ndim == 2) else 'RGB'
        image = Image.fromarray(image, fmt)
        display_image.paste(image, (x, 0))
        x += image.size[1]

    return display_image


def mean_absolute_error(gt: np.ndarray, preds: np.ndarray) -> np.ndarray:
    '''
    Calculates mean absolute error.
    '''
    return np.abs(gt - preds).mean()


def signal_to_noise_ratio(gt: np.ndarray, preds: np.ndarray) -> np.ndarray:
    '''
    Calculates signal-to-noise ratio.
    '''
    return (preds**2).sum() / ((gt - preds)**2).sum()
