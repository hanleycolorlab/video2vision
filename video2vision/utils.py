from contextlib import contextmanager
import csv
from math import floor, isnan
from statistics import mean
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    'extract_samples', 'locate_aruco_markers', 'read_jazirrad_file'
]


@contextmanager
def _coerce_to_2dim(x: Dict):
    '''
    This provides a context inside of which the image in a data dictionary has
    its spatial and (if present) temporal dimensions flattened into a single
    dimension:

    .. code-block:: python

        x = {'image': np.arange(96).reshape(4, 4, 2, 3)}

        with _coerce_to_2dim(x):
            print(x.shape)  # returns (32, 3)

        print(x.shape)  # returns (4, 4, 2, 3)

    This is typically used for math operations, such as linear algebra
    operations, that expect 1- or 2-dimensional inputs.
    '''
    *head, c = x['image'].shape
    x['image'] = x['image'].reshape(-1, c)
    if 'mask' in x:
        x['mask'] = x['mask'].flatten()
    yield
    x['image'] = x['image'].reshape(*head, -1)
    if 'mask' in x:
        x['mask'] = x['mask'].reshape(*head[:2])


@contextmanager
def _coerce_to_3dim(x: Dict):
    '''
    This provides a context inside of which the image in a data dictionary has
    its channel and (if present) temporal dimensions flattened into a single
    dimension:

    .. code-block:: python

        x = {'image': np.arange(96).reshape(4, 4, 2, 3)

        with _coerce_to_3dim(x):
            print(x.shape)  # returns (4, 4, 6)

        print(x.shape)  # returns (4, 4, 2, 3)

    This is typically used for vision operations that expect 3-dimensional
    inputs.
    '''
    tail = x['image'].shape[2:]
    x['image'] = x['image'].reshape(*x['image'].shape[:2], -1)
    yield
    x['image'] = x['image'].reshape(*x['image'].shape[:2], *tail)


@contextmanager
def _coerce_to_4dim(x: Dict):
    '''
    This provides a context inside of which the image in a data dictionary is
    coerced to (H, W, T, C) arrangement, adding a temporal dimension if
    necessary.

    .. code-block:: python

        x = {'image': np.arange(48).reshape(4, 4, 3)}

        with _coerce_to_2dim(x):
            print(x.shape)  # returns (4, 4, 1, 3)

        print(x.shape)  # returns (4, 4, 3)

    This is typically used for operations where we want to be able to index
    over time without worrying about whether time exists to be indexed over.
    '''
    has_t, shape = (x['image'].ndim == 4), x['image'].shape
    if not has_t:
        x['image'] = x['image'].reshape(*shape[:2], 1, shape[2])
    yield
    if not has_t:
        x['image'] = x['image'].squeeze(2)


def _coerce_to_dict(x: Any) -> Dict:
    '''
    This coerces an arbitrary object into a dictionary suitable for use in an
    :class:`Operator`.
    '''
    if not isinstance(x, dict):
        return {'image': _coerce_to_image(x)}
    x['image'] = _coerce_to_image(x['image'])
    if 'mask' in x:
        x['mask'] = _coerce_to_mask(x['mask'])
    return x


def _coerce_to_image(image: Any) -> np.ndarray:
    '''
    Convenience function to coerce an arbitrary objects to a
    :class:`numpy.ndarray` that is either 3- or 4-dimensional.
    '''
    image = np.array(image)
    # Assume a 2-dimensional image is an image with a single band.
    if image.ndim == 2:
        image = image.reshape(*image.shape, 1)
    elif image.ndim not in {3, 4}:
        raise ValueError(
            f'Image must be 3- or 4-dimensional, not {image.shape}'
        )
    if (image.dtype != np.float32):
        image = image.astype(np.float32)
    return image


def _coerce_to_mask(mask: Any) -> np.ndarray:
    '''
    Convenience function to coerce an arbitrary objects to a mask.
    '''
    mask = np.array(mask)
    # Assume a 2-dimensional image is an image with a single band.
    if mask.ndim != 2:
        raise ValueError(
            f'Mask must be 2-dimensional, not {mask.shape}'
        )
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8)
    elif mask.dtype != np.uint8:
        raise ValueError(f'mask dtype {mask.dtype} not supported')
    return mask


def extract_samples(image: np.ndarray, points: np.ndarray, width: int = 10) \
        -> np.ndarray:
    '''
    Extracts samples surrounding sample points from an image.

    Args:
        image (:class:`numpy.ndarray`): Image to extract samples from.
        points (:class:`numpy.ndarray`): Points to take samples from. These
        should be either (x, y) points if the image is an image, or (x, y, t)
        if it is a video.
        width (int): The width of the sample regions.
    '''
    image = _coerce_to_image(image)
    points = np.array(points).astype(np.int64)
    samples = np.empty((points.shape[0], image.shape[-1]), dtype=np.float32)

    min_pts, max_pts = points.min(0), points.max(0)
    (h, w, *t) = image.shape[:-1]
    if (min_pts < 0).any() or (max_pts >= (w, h, *t)).any():
        raise ValueError(
            f'Sample point outside of bounds: {points} vs {image.shape}'
        )
    if (image.ndim == 4) != (points.shape[1] == 3):
        raise ValueError(
            f'Mismatch between image shape and number of point coordinates: '
            f'{image.shape} vs {points}'
        )

    # Convert from center_x, center_y, (t) to l_x, u_y, r_x, d_y, (t).
    r = floor(width / 2)
    points = np.concatenate((
        np.maximum(points[:, 0:2] - r, 0),
        points[:, 0:2] + (width - r),
        points[:, 2:]
    ), axis=1)

    if image.ndim == 3:
        for i, (x_1, y_1, x_2, y_2) in enumerate(points):
            samples[i, :] = image[y_1:y_2, x_1:x_2, :].mean((0, 1))
    elif image.ndim == 4:
        for i, (x_1, y_1, x_2, y_2, t) in enumerate(points):
            samples[i, :] = image[y_1:y_2, x_1:x_2, t, :].mean((0, 1))

    return samples


def _evaluate_ecc_for_warp(template: Dict, image: Dict) -> float:
    '''
    Evaluates a warp by applying it to a stack of images then calculating the
    mean ECC of the warped images against a set of templates. This is a
    convenience wrapper around the function :func:`cv2.computeECC`, which only
    takes one grayscale image at a time.

    Args:
        template (dict): Template images.
        image (dict): Images to match against template images.
    '''
    with _coerce_to_4dim(template), _coerce_to_4dim(image):
        # TODO: Check template for mask also?
        mask = image.get('mask')
        image = image['image'].mean(-1, dtype=np.float32)
        template = template['image'].mean(-1, dtype=np.float32)

        eccs = []
        for t in range(template.shape[2]):
            ecc = cv2.computeECC(
                template[:, :, t],
                image[:, :, t],
                mask,
            )
            # This computation can return NaN in rare cases, e.g. if the image
            # is entirely one value.
            if not isnan(ecc):
                eccs.append(ecc)

    return mean(eccs)


def locate_aruco_markers(x: Dict, marker_ids: Optional[np.ndarray] = None):
    '''
    Convenience function for locating ARUCO markers in a batch of images.

    Something to be aware of: the ARUCO library is not able to detect flipped
    markers.

    Args:
        x (dict): The data dictionary to locate marker IDs in.
        marker_ids (optional, :class:`numpy.ndarray`): If provided, look for
        these specific marker IDs.

    Returns:
        ts (:class:`numpy.ndarray`): The time index for the frames in which the
        markers were found.
        corners: The point locations of the corners. If marker_ids is provided,
        this is returned in shape (time, marker_id, corner_id, x or y). If it
        is not, this is a list of :class:`numpy.ndarray`, each entry in shape
        (marker_id, corner_id, x or y).
        marker_ids (list of :class:`numpy.ndarray`): This is only returned if
        marker_ids is not provided. It contains the IDs of the markers.
    '''
    x = _coerce_to_dict(x)
    ts, corners, found_ids = [], [], []

    detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.adaptiveThreshWinSizeMax = 256

    with _coerce_to_4dim(x):
        w = x['image'].shape[1]
        # ARUCO detector requires uint8, but we now work in float32.
        image = np.clip(x['image'] * 256., 0, 255).astype(np.uint8)

        for t in range(x['image'].shape[2]):
            pts, ids, _ = cv2.aruco.detectMarkers(
                image[:, :, t, :],
                _MARKER_DICTIONARY,
                parameters=detector_params,
            )
            # The ARUCO detector checks for rotated markers, but not flipped.
            # So we do it manually.
            if ids is None:
                pts, ids, _ = cv2.aruco.detectMarkers(
                    image[:, ::-1, t, :],
                    _MARKER_DICTIONARY,
                    parameters=detector_params,
                )
                for pt in pts:
                    pt[:, :, 0] = w - pt[:, :, 0]
            if ids is not None:
                ts.append(t)
                pts = np.stack([box.reshape(4, 2) for box in pts], axis=0)
                corners.append(pts)
                found_ids.append(list(ids.flatten()))

    if not ts:
        raise RuntimeError('Failed to locate markers')

    if marker_ids is not None:
        try:
            ts, corners, found_ids = zip(*(
                (t, b, ids) for (t, b, ids) in zip(ts, corners, found_ids)
                if set(ids) == set(marker_ids)
            ))
        except ValueError:
            raise RuntimeError('Failed to locate markers')

        ts = np.array(ts)
        corners = [
            np.array([c[ids.index(i)] for i in marker_ids])
            for c, ids in zip(corners, found_ids)
        ]
        corners = np.stack(corners, axis=0)
        return ts, corners

    else:
        found_ids = [np.array(ids) for ids in found_ids]
        return np.array(ts), corners, found_ids


def read_jazirrad_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reads a jazirrad file containing irradiance data, and returns a pair of
    :class:`numpy.ndarray`, one containing the wavelengths and the other
    containing the irradiance.

    Args:
        path (str): Path to the file.
    '''
    with open(path, 'r') as irrad_file:
        contents = irrad_file.readlines()

    if contents[0] != 'Jaz Absolute Irradiance File\n':
        raise ValueError('File does not match Jaz Irradiance format')

    # These lines delineate the tab-separated values data containing the
    # irradiances. There's also a header we skip.
    start_idx = contents.index('>>>>>Begin Processed Spectral Data<<<<<\n') + 2
    end_idx = contents.index('>>>>>End Processed Spectral Data<<<<<\n')
    contents = contents[start_idx:end_idx]
    # contents is now a tab-separated with four columns. The 0th is wavelength
    # and the second is the spectral response in uW / cm^2 nm.
    contents = list(csv.reader(contents, dialect='excel-tab'))
    wavelengths, _, response, _ = zip(*contents)
    wavelengths = np.array([float(w) for w in wavelengths], dtype=np.float32)
    response = np.array([float(r) for r in response], dtype=np.float32)

    return wavelengths, response


class Registry:
    '''
    This provides a convenient way to create global dictionaries that map names
    to object classes. It's inspired by:

        https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/registry.py
    '''
    def __init__(self):
        self._obj_name: Dict[str, Any] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._obj_name

    def _do_register(self, name: str, obj: Any):
        if name in self._obj_name:
            raise ValueError(f'{name} already registered')
        self._obj_name[name] = obj

    def get(self, name: str) -> Any:
        if name not in self._obj_name:
            raise KeyError(f'{name} not found in registry')
        return self._obj_name[name]

    def register(self, obj: Any = None):
        # In this case, it's used as a decorator.
        if obj is None:
            def deco(cls: Any):
                self._do_register(cls.__name__, cls)
                return cls
            return deco

        else:
            self._do_register(obj.__name__, obj)
            return obj


_MARKER_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
