import json
import os
from typing import Optional

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
    Calculates coefficient of determination. This calculates the ``biologist's
    $R^2$'', where we apply a linear regression before calculating the sum of
    residuals and sum of total variation. So, for example, if we had ground
    truth $x$ and prediction $\\hat{x} = -x$, the coefficient of determination
    would be 1.0.
    '''
    pred_sq_sum, pred_sum = preds.dot(preds), preds.sum()
    gram_matrix = np.array([
        [pred_sq_sum, pred_sum],
        [pred_sum, preds.shape[0]]
    ])
    moment_matrix = np.array([preds.dot(gt), gt.sum()])
    # A LinAlgError can occur if the gram_matrix is singular, in which case we
    # drop the constant term
    try:
        beta_hat = np.linalg.inv(gram_matrix).dot(moment_matrix)
    except np.linalg.LinAlgError:
        if pred_sq_sum:
            beta_hat = np.array([moment_matrix[0] / pred_sq_sum, 0])
        else:
            beta_hat = np.array([0, 0])
    fit_preds = beta_hat[0] * preds + beta_hat[1]
    ss_res, ss_tot = ((gt - fit_preds)**2).sum(), ((gt - gt.mean())**2).sum()

    if ss_tot == ss_res == 0:
        return 1
    elif ss_tot == 0:
        return 0
    else:
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
