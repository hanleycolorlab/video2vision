from contextlib import contextmanager
from copy import copy
import csv
from math import exp, floor, isnan, log2
import os
from statistics import mean
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import scipy

_CV2_VERSION = tuple(int(x) for x in cv2.__version__.split('.'))

__all__ = [
    'detect_motion', 'extract_audio_from_mp4', 'extract_samples',
    'get_photoreceptor_template', 'get_temporal_offset_from_audio',
    'locate_aruco_markers', 'read_jazirrad_file'
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
    if not isinstance(image, np.ndarray):
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


def detect_motion(image: Dict, area_threshold: float = 0.005,
                  difference_threshold: float = 0.1,
                  background: Optional[Dict] = None) -> np.ndarray:
    '''
    Detects motion in a stack of frames, returning a boolean mask of which
    frames contain movement. The algorithm works by looking for changes from
    background in absolute magnitude:

        1. Convert the image to greyscale.
        2. Apply a Gaussian blur.
        3. Calculate the absolute difference in pixel values between the image
           and a background frame.
        4. Calculate the percentage of pixels where the absolute difference
           exceeds a specified threshold.
        5. The image is considered to show motion if the percentage exceeds an
           area threshold.

    If the background is not supplied, it is calculated by averaging the frames
    in the stack.

    Args:
        image (dict): The images to detect motion in.
        area_threshold (float): Threshold for the minimum percentage of area
        that must show an absolute difference exceeding the threshold in order
        to declare motion.
        difference_threshold (float): Threshold for the absolute difference to
        declare a pixel as showing motion.
        background (optional, dict): The background.
    '''
    with _coerce_to_4dim(image):
        # Find mask. We will avoid checking for motion in areas which are
        # masked out.
        mask = image.get('mask', None)
        if (background is not None) and 'mask' in background:
            if mask is None:
                mask = background['mask']
            else:
                mask &= background['mask']

        if background is None:
            background = _prep_for_motion_detection(_extract_background(image))

        # images will be in shape HWT1 after preparation.
        image = _prep_for_motion_detection(image['image'])

        motion = []
        with _coerce_to_4dim(background):
            # Confirm background has only one frame
            if background['image'].shape[2] != 1:
                raise ValueError(
                    f"Background has shape {background['image'].shape}"
                )

            for t in range(image['image'].shape[2]):
                diff = cv2.absdiff(
                    image['image'][:, :, t, 0],
                    background['image'][:, :, 0, 0]
                )
                if mask is not None:
                    # mask is of type uint8, and needs to be cast or it will be
                    # interpreted as indices instead of a mask
                    diff = diff[mask.astype(np.bool_)]
                diff = (diff > difference_threshold).mean()
                motion.append(diff > area_threshold)

    return np.array(motion)


def extract_audio_from_mp4(path: str, sample_rate_per_frame: int = 1) \
        -> np.ndarray:
    '''
    Extracts the audio from an MP4 file on disk and returns it as a
    :class:`numpy.ndarray`.

    Args:
        path (str): Path to the MP4 file.
        sample_rate_per_frame (int): Audio is not stored at the same sample
            rate as video - essentially, it has more "frames" than the video
            has. The audio will be resampled to approximately this many audio
            samples per video frame. The resampling is only approximate,
            however.

    Returns:
        A 2-dimensional :class:`numpy.ndarray`, with the 0th dimension indexing
        the time and the 1st dimension indexing the channel. There will
        ordinarily be two channels.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mp4_fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
    ar = int(mp4_fps * sample_rate_per_frame)

    with tempfile.TemporaryDirectory() as temp_root:
        wav_path = os.path.join(temp_root, 'audio.wav')
        # Flag meanings:
        # ab: Audio bit rate
        # ac: Number of audio channels
        # ar: Audio out rate
        proc = subprocess.run(
            f'ffmpeg -i {path} -ab 160k -ac 2 -ar {ar} -vn {wav_path}',
            shell=True,
            capture_output=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode())

        _, audio = scipy.io.wavfile.read(wav_path)

    return audio


def _extract_background(image: Dict) -> Dict:
    '''
    Estimates the backgroud across time of a batch of images.
    '''
    # We're going to modify the values of the dictionary, so ensure it won't
    # flow back upstream.
    image = copy(_coerce_to_dict(image))

    with _coerce_to_4dim(image):
        # For small time dimensions, the median is less stable than the mean.
        if image['image'].shape[2] >= 8:
            image['image'] = np.median(image['image'], axis=2, keepdims=True)
        else:
            image['image'] = np.mean(image['image'], axis=2, keepdims=True)

    return image


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
    # TODO: Refactor image to dict
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


def get_photoreceptor_template(peak: float,
                               wavelengths: np.ndarray = np.arange(300, 701),
                               template: str = 'A1') -> np.ndarray:
    '''
    Convenience function for calculating the photoreceptor sensitivities for
    the templates from:

        Govardovskii, et al. 2000. ``In Search of the Visual Pigment
        Template.'' Visual Neuroscience, Vol. 17 No. 4, pp. 509-528.

    Args:
        peak (float): Peak sensitivity of the photoreceptor, in nm.
        wavelengths (:class:`numpy.ndarray`): The wavelengths to calculate the
        photoreceptor sensitivities from, in nm. Default: evenly spaced every 1
        nm from 300 to 700 nm inclusive.
        template (str): Which template to use. Choices: 'a1', 'a2'.
    '''
    x = peak / wavelengths

    if template == 'A1':
        a = 0.8795 + 0.0459 * exp(-(peak - 300) ** 2 / 11940)
        alpha = 1 / (
            np.exp(69.7 * (a - x)) + np.exp(28 * (0.922 - x)) +
            np.exp(-14.9 * (1.104 - x)) + 0.674
        )
        b = -40.5 + 0.195 * peak
        beta = 0.26 * np.exp(-((wavelengths - (189 + 0.315 * peak)) / b) ** 2)

    elif template == 'A2':
        A = 62.7 + 1.834 * exp((peak - 625) / 54.2)
        a = 0.875 + 0.0268 * exp((peak - 665) / 40.7)
        alpha = 1 / (
            np.exp(A * (a - x)) + np.exp(20.85 * (0.9101 - x)) +
            np.exp(-10.37 * (1.1123 - x) + 0.5343)
        )
        b = 317 - 1.149 * peak + 0.00124 * peak**2
        beta = 0.37 * np.exp(-((wavelengths - (216.7 + 0.287 * peak)) / b)**2)

    else:
        raise ValueError(template)

    sensitivity = alpha + beta
    return sensitivity / sensitivity.sum()


def get_temporal_offset_from_audio(audio_1: np.ndarray, audio_2: np.ndarray) \
        -> int:
    '''
    Given two single-channel streams of audio, estimates the offset between
    them in time steps using cross-correlation.
    '''
    if {audio_1.ndim, audio_2.ndim} != {1}:
        raise ValueError(
            f'Received malformed arrays: {audio_1.shape}, {audio_2.shape}'
        )

    n_t = min(audio_1.shape[0], audio_2.shape[0])
    n_t = 2 ** floor(log2(n_t))
    audio_1, audio_2 = audio_1[:n_t], audio_2[:n_t]
    corr = scipy.fft.ifft(
        scipy.fft.fft(audio_1) * np.conj(scipy.fft.fft(audio_2))
    )
    max_corr = np.argmax(np.absolute(corr))

    return (max_corr - n_t) if (max_corr > n_t // 2) else max_corr


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

    # Function name changed in 4.7.0
    if _CV2_VERSION >= (4, 7, 0):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.adaptiveThreshWinSizeMax = 256

    with _coerce_to_4dim(x):
        # ARUCO detector requires uint8, but we now work in float32. It also
        # requires that the image be either 1- or 3-channel, so we convert to
        # greyscale.
        image = x['image'].mean(-1, keepdims=True, dtype=np.float32)
        image = np.clip(image * 256., 0, 255).astype(np.uint8)

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
                    pt[:, :, 0] = x['image'].shape[1] - pt[:, :, 0]
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


def _prep_for_motion_detection(image: Dict) -> Dict:
    '''
    Applies pre-processing steps prior to motion detection.
    '''
    # We're going to modify the values of the dictionary, so ensure it won't
    # flow back upstream.
    image = copy(_coerce_to_dict(image))

    # Convert to greyscale. We need to make sure we end up with a new array out
    # of this, as we're going to modify image['image'] in-place after this.
    if image['image'].shape[-1] == 3:
        with _coerce_to_2dim(image):
            # This function requires an input of shape HW3, and returns an
            # output of shape HW. We convert the image to shape (HWT)13, so it
            # returns an output of shape (HWT)1, which then becomes HWT1 when
            # we exit the context.
            image['image'] = cv2.cvtColor(
                image['image'].reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
            )
    elif image['image'].shape[-1] == 1:
        image['image'] = image['image'].copy()
    else:
        image['image'] = image['image'].mean(2, keepdims=True)

    # Apply Gaussian blur in place.
    with _coerce_to_4dim(image):
        # We want to avoid a copy from a stack. However, cv2.GaussianBlur
        # requires that its out argument must be in contiguous arrangement.
        dst = np.empty(
            tuple(image['image'].shape[i] for i in [2, 0, 1, 3]),
            dtype=np.float32
        )
        for t in range(image['image'].shape[2]):
            cv2.GaussianBlur(
                image['image'][:, :, t, :],
                dst=dst[t, :, :, :],
                ksize=(5, 5),
                sigmaX=0,
            )
        image['image'] = np.rollaxis(dst, 0, 3)

    return image


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
    try:
        start_idx = contents.index(
            '>>>>>Begin Processed Spectral Data<<<<<\n'
        )
    except ValueError:
        start_idx = contents.index('>>>>>Begin Spectral Data<<<<<\n')
    try:
        end_idx = contents.index('>>>>>End Processed Spectral Data<<<<<\n')
    except ValueError:
        end_idx = contents.index('>>>>>End Spectral Data<<<<<\n')
    contents = contents[start_idx + 2:end_idx]
    # contents is now a tab-separated with four columns. The 0th is wavelength
    # and the second is the spectral response in uW / cm^2 nm.
    contents = list(csv.reader(contents, dialect='excel-tab'))
    wavelengths, _, _, response = zip(*contents)
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
