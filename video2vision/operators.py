'''
This provides various :class:`Operator` s, which represent operations performed
on an image. We build them as distinct classes so that they can be saved to
keep track of what operations were performed to produce an image, and reused on
other images.
'''
from copy import copy
import json
from math import sqrt
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tdigest

from .utils import (
    _coerce_to_2dim,
    _coerce_to_3dim,
    _coerce_to_4dim,
    _coerce_to_dict,
    Registry,
)

__all__ = [
    'ConcatenateOnBands', 'HistogramStretch', 'HorizontalFlip', 'LinearMap',
    'load_operator', 'Operator', 'OPERATOR_REGISTRY', 'Pad', 'ResetPipeline',
    'Resize', 'ToRNL', 'UBGRtoXYZ', 'VerticalFlip',
]

# This provides a registry of operators. This is used when saving and restoring
# Operators, since it allows us to map from operator name to the operator
# class.
OPERATOR_REGISTRY = Registry()


class HoldToken:
    '''
    This is a dummy class returned by an :class:`Operator` to indicate that the
    operator is not ready to proceed. This is usually used by
    :class:`AutoOperator` s that have not yet received enough input to deterine
    their coefficients. All downstream operators then return a
    :class:`HoldToken` as well.
    '''


class ResetPipeline(Exception):
    '''
    This exception should be raised by :class:`Operator` s that want to reset
    the :class:`Pipeline` and reprocess all inputs. This is generally used by
    :class:`AutoOperator` s that need multiple batches to determine their
    correct coefficients.
    '''


class Operator:
    '''
    This is the base class for an operator. To add a new operator, subclass
    this class, follow the API described below, and register the operator with:

    .. code-block:: python

        import video2vision as v2v

        v2v.OPERATOR_REGISTRY.register
        class NewOperator(Operator):
            ...

    All subclasses of :class:`Operator` must provide the following methods or
    attributes:

        * __init__: All arguments must be able to be taken in a form that is
          JSON-serializable. For example, if one of the arguments is expected
          to be a :class:`numpy.ndarray`, the method must also except a list of
          floats, and coerce that list.
        * apply: This should take in one or more images or videos without any
          other arguments, apply the operator, then return the output. The
          inputs are provided as dictionaries with the following key-value
          pairs:

             - 'image': Contains a :class:`numpy.ndarray` of dtype float32 that
               is either 3-dimensional (in H, W, C arrangement) or
               4-dimensional (in H, W, T, C arrangement). This is a mandatory
               key. The image must be scaled to the range [0, 1].
             - 'mask': Contains a :class:`numpy-ndarray` of dtype uint8 that is
               2-dimensional (in H, W arrangement) that provides a boolean mask
               of which pixels are valid. This is an optional key; if not
               present, all pixels are considered valid. Can also be None, to
               denote all pixels valid.

          There should be a single output, consisting of a dictionary with the
          same structure. None of the input :class:`numpy.ndarray` s should be
          modified in-place, since they may be being used as inputs to multiple
          :class:`Operator` s. However, the dictionary may be modified in-
          place.
        * apply_points: This should map a 2-dimensional :class:`numpy.ndarray`
          of points in an input image to points in an output image. If this
          cannot be consistently done, it should raise a RuntimeError.
        * _to_json: This should serialize the operator to JSON format. The
          output should be a dictionary with the key 'class', whose value is
          the name of the operator class as a string. All other key-value pairs
          should be the keyword arguments passed to the operator when it is
          instantiated, coerced to forms that can be JSON-serialized. For
          example, a :class:`numpy.ndarray` must have the :meth:`tolist` method
          called.
        * num_inputs: This should be an attribute specifying the number of
          inputs that are expected by the apply method. If the value is -1,
          then the :class:`Operator` takes a variable number of inputs.

    All operators can be applied to either an image or a video. Images are
    expected to be a :class:`numpy.ndarray` in (H, W, C) arrangement, where H
    indexs height, W indexs width, and C indexs channel. Videos are expected to
    be a :class:`numpy.ndarray` in (H, W, T, C) arrangement, where T indexs
    time. Note that one consequence of this is that we need to be careful in
    our indexing.
    '''
    num_inputs = 1

    def __call__(self, *xs) -> Union[Dict, HoldToken]:
        if any(isinstance(x, HoldToken) for x in xs):
            return HoldToken()

        # copy performs a shallow copy. That is, if x is a dictionary, it
        # creates a new dictionary, but the values are all the same as the
        # original values. This avoids copying images - which would be wasteful
        # of both memory and compute - but allows the apply method to modify
        # the dictionary in-place.
        xs = tuple(copy(_coerce_to_dict(x)) for x in xs)
        return self.apply(*xs)

    def __repr__(self) -> str:
        desc = self._to_json()
        name = desc.pop('class')
        return f"{name}({', '.join(f'{k}={v}' for k, v in desc.items())})"

    def _to_json(self) -> Dict:
        return {'class': self.__class__.__name__}

    def apply(self, *xs) -> Dict:
        raise NotImplementedError()

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        raise RuntimeError()

    def release(self):
        '''
        This method is used to notify an :class:`Operator` that the pipeline
        has finished processing and any cleanup operations that need to be
        performed should take place.
        '''
        return

    def reset(self):
        '''
        This method is used to notify an :class:`Operator` that the pipeline is
        being reset to the beginning. This is usually used when
        :class:`AutoOperator` s need to process multiple batches before
        determining their coefficients. This does not indicate a hard reset:
        an :class:`AutoOperator` that *has* determined its coefficients should
        not discard them.
        '''
        return

    def set_batch_size(self, batch_size: int):
        '''
        This method is used to notify an :class:`Operator` what the expected
        number of frames per batch will be. It is not guaranteed that the
        :class:`Operator` will receive that exact number of frames, merely that
        that is the expected number.
        '''
        return


@OPERATOR_REGISTRY.register
class ConcatenateOnBands(Operator):
    '''
    This concatenates multiple images together, subselecting the bands.
    '''
    def __init__(self, bands: List[List[int]]):
        '''
        Args:
            bands (list of list of int): The bands to use.
        '''
        self.bands = bands

    def apply(self, *xs) -> Dict:
        if self.bands is None:
            self.bands = [list(range(x['image'].shape[-1])) for x in xs]

        if len(xs) != len(self.bands):
            raise ValueError(
                f'Mismatch between number of input images and merge: '
                f'{len(xs)} vs {len(self.bands)}'
            )

        # Construct holder for return value. rtn['image'] will initially be a
        # List[np.ndarray], which we will concatenate at the end.
        rtn = {'image': [], 'names': []}

        for x, bands_from_image in zip(xs, self.bands):
            # If paths is in there, delete it, because it will no longer be
            # valid
            x.pop('paths', None)
            image = x.pop('image')
            for band in bands_from_image:
                if band >= image.shape[-1]:
                    raise ValueError(
                        f'Band {band} not in image of shape {image.shape}'
                    )
                rtn['image'].append(image[..., band])
            # Combine masks through intersection
            if 'mask' in x:
                if 'mask' in rtn:
                    rtn['mask'] &= x.pop('mask')
                else:
                    rtn['mask'] = x.pop('mask')
            # Combine names
            if 'names' in x:
                rtn['names'].append(x.pop('names'))
            # Append any other keys
            for k, v in x.items():
                if k not in rtn:
                    rtn[k] = v
                elif k == 'final':
                    rtn[k] = rtn[k] or v
                elif rtn[k] != v:
                    raise ValueError(
                        f'Conflicting values for key {k} in concatenated '
                        f'images: {v} vs.{rtn[k]}'
                    )

        rtn['image'] = np.stack(rtn['image'], axis=-1)

        if rtn['names']:
            rtn['names'] = ['-'.join(x) for x in zip(*rtn['names'])]
        else:
            rtn.pop('names')

        return rtn

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'bands': self.bands,
        }

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    @property
    def num_inputs(self) -> int:
        return len(self.bands)

    @property
    def num_out_bands(self) -> int:
        '''
        Number of output bands.
        '''
        return sum(len(b) for b in self.bands)


@OPERATOR_REGISTRY.register
class HistogramStretch(Operator):
    '''
    Applies a histogram stretch to the image. A given percentile $k$ is
    specified by the user, and the $k$th and $(1 - k)$th percentiles of the
    brightness are calculated. Pixels above the $(1 - k)$th percentile or below
    the $k$th percentile are squashed to 1 and 0 respectively, while pixels in
    between are scaled so that the range of brightness lies between 0 and 1.
    '''
    def __init__(self, perc: float, apply_per_frame: bool = True,
                 ceiling: Optional[float] = None,
                 floor: Optional[float] = None):
        '''
        Args:
            perc (float): The percentile to use in the stretch. This is
            expected to be in the range [0, 100].
            apply_per_frame (bool): If true, the histogram stretch is applied
            individually per frame. If false, the percentiles are calculated
            over the entire stack of images, then applied uniformly. True is
            generally appropriate for still images, while false is generally
            appropriate for videos.
            ceiling (optional, float): This can be used to supply the $(1 -
            k)$th percentile if it is already known. This is ignored if
            apply_per_frame is true.
            floor (optional, float): This can be used to supply the $k$th
            percentile if it is already known. This is ignored if
            apply_per_frame is true.
        '''
        self.perc = perc
        self.apply_per_frame = apply_per_frame
        self.ceiling = ceiling
        self.floor = floor
        # This is used when calculating the percentiles over an entire stack of
        # frames, e.g. when apply_per_frame is false and we have not yet
        # determined ceiling and floor
        self.digest = None if apply_per_frame else tdigest.TDigest()

    def apply(self, x: Dict) -> Dict:
        # This has entirely different behavior when applied per frame or not.
        with _coerce_to_4dim(x):
            if self.apply_per_frame:
                floor, ceiling = np.percentile(
                    x['image'], [self.perc, 100 - self.perc], axis=(0, 1, 3),
                    keepdims=True,
                )
            elif self.ceiling is not None:
                floor, ceiling = self.floor, self.ceiling
            else:
                self.digest.batch_update(x['image'].flatten())
                if x.get('final', False):
                    self.floor = floor = self.digest.percentile(self.perc)
                    self.ceiling = ceiling = self.digest.percentile(
                        100 - self.perc
                    )
                    raise ResetPipeline()
                else:
                    return HoldToken()

            x['image'] = (x['image'] - floor) / (ceiling - floor)
            x['image'] = np.clip(x['image'], 0, 1)

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'perc': self.perc,
            'apply_per_frame': self.apply_per_frame,
            'ceiling': self.ceiling,
            'floor': self.floor,
        }


@OPERATOR_REGISTRY.register
class HorizontalFlip(Operator):
    '''
    This flips the image horizontally.
    '''
    def apply(self, x: Dict) -> Dict:
        x['image'] = x['image'][:, ::-1]
        if x.get('mask') is not None:
            x['mask'] = x['mask'][:, ::-1]
        return x


@OPERATOR_REGISTRY.register
class LinearMap(Operator):
    '''
    Applies a linear map to an image:

    .. code-block:: python

        out[y, x, band] = sum(image[y, x, :] * mat[:, band])
    '''
    def __init__(self, mat: np.ndarray):
        '''
        Args:
            mat (:class:`numpy.ndarray`): Matrix defining the linear map.
        '''
        self.mat = np.array(mat).astype(np.float32)

    def apply(self, x: Dict) -> Dict:
        if x['image'].shape[-1] != self.mat.shape[0]:
            raise ValueError(
                f"Mismatch between number of image bands and map: "
                f"{x['image'].shape[0]} vs. {self.mat.shape[0]}"
            )

        x['image'] = np.dot(x['image'], self.mat)

        return x

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'mat': self.mat.tolist(),
        }

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    def apply_values(self, values: np.ndarray) -> np.ndarray:
        '''
        This is a convenience function that applies the functions to a set of
        values of shape (N, B), where N indexs the value and B indexs the band.
        '''
        return np.dot(values, self.mat)

    @classmethod
    def build_sensor_convertor(cls, reflectances: np.ndarray,
                               source_sensitivities: np.ndarray,
                               control_sensitivities: np.ndarray,
                               source_illum: Optional[np.ndarray] = None,
                               control_illum: Optional[np.ndarray] = None):
        '''
        Builds a :class:`LinearMap` for converting from one sensor to another
        sensor.

        Args:
            reflectances (:class:`np.ndarray`): Expected to be a 2-dimensional
            array, with the 0th dimension indexing the wavelength of the
            incident radiation and the 1st dimension indexing the material.
            source_sensitivities (:class:`np.ndarray`): Expected to be a 2-
            dimensional array, with the 0th dimension indexing the wavelength
            and the 1st dimension indexing the band.
            control_sensitivities (:class:`np.ndarray`): Expected to be a 2-
            dimensional array, with the 0th dimension indexing the wavelength
            and the 1st dimension indexing the band.
            source_illum (optional, :class:`np.ndarray`): Expected to be a 1-
            dimensional array, indexed by wavelength.
            control_illum (optional, :class:`np.ndarray`): Expected to be a 1-
            dimensional array, indexed by wavelength.
        '''
        source_illum = 1 if (source_illum is None) else source_illum
        control_illum = 1 if (control_illum is None) else control_illum

        # Predicted received QC for source and control sensor from reflectances
        source_qc = reflectances.T.dot(source_illum * source_sensitivities)
        control_qc = reflectances.T.dot(control_illum * control_sensitivities)
        # rcond must be specified to suppress a warning
        mat, *_ = np.linalg.lstsq(source_qc, control_qc, rcond=None)

        return cls(mat)


@OPERATOR_REGISTRY.register
class Pad(Operator):
    '''
    This pads an image.
    '''
    def __init__(self, pad: Tuple[int, int, int, int],
                 value: Union[int, float]):
        '''
        Args:
            pad (tuple of int): Values to pad on top, bottom, left, and right.

            value (int or float): Value to fill the padded region with.
        '''
        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        self.pad = pad
        self.value = value

    def apply(self, x: Dict) -> Dict:
        with _coerce_to_3dim(x):
            if x.get('mask') is None:
                x['mask'] = np.ones(x['image'].shape[:2], dtype=np.uint8)
            x['image'] = cv2.copyMakeBorder(
                x['image'], *self.pad, cv2.BORDER_CONSTANT, value=self.value
            )
            x['mask'] = cv2.copyMakeBorder(
                x['mask'], *self.pad, cv2.BORDER_CONSTANT, value=0
            )

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        shift = np.array([[self.pad[2], self.pad[0]]])
        return pts + shift

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'pad': self.pad,
            'value': self.value,
        }


@OPERATOR_REGISTRY.register
class Resize(Operator):
    '''
    This class wraps the :func:`cv2.resize` function.
    '''
    def __init__(self, scale: Union[float, Tuple[int, int]],
                 sampling_mode: int = cv2.INTER_LINEAR):
        '''
        Args:
            scale (float or tuple of int): This is either a float, giving the
            proportional scale, or a pair of ints, giving the desired output
            size in pixels.

            sampling_mode (int): Method of interpolation.
        '''
        self.scale = scale
        self.sampling_mode = sampling_mode

    def apply(self, x: Dict) -> Dict:
        if isinstance(self.scale, float):
            out_h = int(self.scale * x['image'].shape[0])
            out_w = int(self.scale * x['image'].shape[1])
        else:
            out_w, out_h = self.scale

        # If this is a video, we convert it to (H, W, C * T) arrangement. cv2
        # will treat this as an image with an unusually large number of
        # channels and be perfectly fine with it.
        with _coerce_to_3dim(x):
            x['image'] = cv2.resize(
                x['image'], (out_w, out_h), interpolation=self.sampling_mode
            )
            if x.get('mask') is not None:
                x['mask'] = cv2.resize(
                    x['mask'], (out_w, out_h), interpolation=cv2.INTER_NEAREST
                )

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        if isinstance(self.scale, float):
            return self.scale * pts
        else:
            raise RuntimeError()

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'scale': self.scale,
            'sampling_mode': self.sampling_mode,
        }


@OPERATOR_REGISTRY.register
class ToRNL(Operator):
    '''
    This operator converts a three- or four-channel image to the RNL color
    space, as given in Appendix A of:

        Renoult et al., 2017. "Colour Spaces in Ecology and Evolutionary
        Biology." *Biological Reviews*, Vol. 92, pp. 292-315.

    $L^1$ distance in the RNL color space should correspond to distance in
    perceptual space.
    '''
    def __init__(self, photo_density: np.ndarray,
                 photo_sensitivity: np.ndarray,
                 background: Union[float, np.ndarray] = 0.5,
                 weber_fraction: float = 0.1,
                 illuminance: Union[float, np.ndarray] = 1.0,
                 normalize_outputs: bool = True,
                 scale_by_magnitude: bool = False):
        '''
        Args:
            photo_density (:class:`numpy.ndarray`): Relative density of
            photoreceptors, in shape (band,).
            photo_sensitivity (:class:`numpy.ndarray`): Photoreceptor
            sensitivity, in shape (wavelength, band).
            background (float or :class:`numpy.ndarray`): Background
            reflectance.
            weber_fraction (float): Weber fraction, denoting the proportion of
            a value that must change for the change to be detectable by the
            organism.
            illuminance (float or :class:`numpy.ndarray`): Illuminance.
            normalize_outputs (bool): Whether to normalize the outputs to the
            range [0, 1]. If this is not set, the outputs will typically be in
            roughly the range [-45, 45].
            scale_by_magnitude (bool): Whether to scale the output values by
            the magnitude of the input values. This ensures that dark inputs
            are mapped to dark outputs.
        '''
        self.photo_density = np.array(photo_density).astype(np.float32)
        self.photo_sensitivity = np.array(photo_sensitivity).astype(np.float32)
        self.scale_by_magnitude = scale_by_magnitude
        self.normalize_outputs = normalize_outputs
        self.weber_fraction = weber_fraction

        if isinstance(background, float):
            self.background = background
        else:
            self.background = np.array(background).astype(np.float32)

        if isinstance(illuminance, float):
            self.illuminance = illuminance
        else:
            self.illuminance = np.array(illuminance).astype(np.float32)

    def apply(self, x: Dict) -> Dict:
        with _coerce_to_2dim(x):
            # Since we're going to apply a log transform, we need this to be
            # positive.
            qc = np.clip(x['image'], 0.5 / 256, None)

            n_bands = {
                self.photo_sensitivity.shape[1], qc.shape[1],
                len(self.photo_density)
            }
            if len(n_bands) > 1:
                raise ValueError('Mismatch in number of bands')

            background = (
                self.background * self.photo_sensitivity * self.illuminance
            )
            background = background.sum(0, keepdims=True)
            s = np.log(qc / background)  # A 1.6
            e = (
                self.weber_fraction *
                np.sqrt(self.photo_density[-1] / self.photo_density)
            )

            if qc.shape[1] == 4:
                sum_of_e1 = e[2]**2 + e[3]**2
                sum_of_e2 = (
                    (e[2] * e[3])**2 + (e[1] * e[2])**2 + (e[1] * e[3])**2
                )
                sum_of_e3 = (
                    (e[1] * e[2] * e[3])**2 + (e[0] * e[2] * e[3])**2 +
                    (e[0] * e[1] * e[3])**2 + (e[0] * e[1] * e[2])**2
                )
                A = sqrt(sum_of_e2 / sum_of_e3)
                a = (e[1] * e[2])**2 / sum_of_e2
                b = (e[1] * e[3])**2 / sum_of_e2
                c = (e[2] * e[3])**2 / sum_of_e2

                out_x = s[:, 3] - s[:, 2] / sqrt(e[2]**2 + e[3]**2)
                out_y = (
                    sqrt(sum_of_e1 / sum_of_e2) *
                    (s[:, 1] - (s[:, 3] * (e[2]**2 / sum_of_e1))
                     - (s[:, 2] * (e[3]**2 / sum_of_e1)))
                )
                out_z = (
                    A * (s[:, 0] - (a * s[:, 3] + b * s[:, 2] + c * s[:, 1]))
                )
                x['image'] = np.stack((out_z, out_y, out_x), axis=1)

                if self.normalize_outputs:
                    min_s = np.log((0.5 / 256) / background.flatten()).min()
                    max_s = np.log(1 / background.flatten()).max()
                    min_out = (
                        min(1 / sqrt(sum_of_e1), sqrt(sum_of_e1 / sum_of_e2),
                            A) * (min_s - max_s)
                    )
                    max_out = (
                        max(1 / sqrt(sum_of_e1), sqrt(sum_of_e1 / sum_of_e2),
                            A) * (max_s - min_s)
                    )
                    x['image'] = (x['image'] - min_out) / (max_out - min_out)

            elif qc.shape[1] == 3:
                sum_of_e1 = e[1]**2 + e[2]**2
                sum_of_e2 = (
                    (e[0] * e[1])**2 + (e[0] * e[2])**2 + (e[1] * e[2])**2
                )

                out_x = sqrt(1 / sum_of_e1) * (s[:, 2] - s[:, 1])
                out_y = (
                    sqrt(sum_of_e1 / sum_of_e2) *
                    (s[:, 0] - (s[:, 2] * e[1]**2 / sum_of_e1)
                     - (s[:, 1] * e[2]**2 / sum_of_e1))
                )
                x['image'] = np.stack((out_y, out_x), axis=1)

                if self.normalize_outputs:
                    min_s = np.log((0.5 / 256) / background.flatten()).min()
                    max_s = np.log(1 / background.flatten()).max()
                    min_out = (
                        min(sqrt(1 / sum_of_e1), sqrt(sum_of_e1 / sum_of_e2))
                        * (min_s - max_s)
                    )
                    max_out = (
                        max(sqrt(1 / sum_of_e1), sqrt(sum_of_e1 / sum_of_e2))
                        * (max_s - min_s)
                    )
                    x['image'] = (x['image'] - min_out) / (max_out - min_out)

            else:
                raise NotImplementedError(qc.shape[1])

            if self.scale_by_magnitude:
                scale = qc.mean(axis=1) / (x['image'].mean(axis=1) + 1e-6)
                x['image'] *= scale.reshape(-1, 1)

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    def _to_json(self) -> Dict:
        if isinstance(self.background, float):
            bg = self.background
        else:
            bg = self.background.tolist()
        if isinstance(self.illuminance, float):
            illum = self.illuminance
        else:
            illum = self.illuminance.tolist()

        return {
            'class': self.__class__.__name__,
            'photo_density': self.photo_density.tolist(),
            'photo_sensitivity': self.photo_sensitivity.tolist(),
            'background': bg,
            'weber_fraction': self.weber_fraction,
            'illuminance': illum,
            'scale_by_magnitude': self.scale_by_magnitude,
        }


@OPERATOR_REGISTRY.register
class UBGRtoXYZ(Operator):
    '''
    This operator converts a four-channel image containing ultraviolet, blue,
    green, and red bands to an easily visualizable 3-channel form. It expects
    to be passed four bands, in that order. The equations are given by:

    .. math::

        u = \\frac{U}{U + B + G + R}
        s = \\frac{B}{U + B + G + R}
        m = \\frac{G}{U + B + G + R}
        l = \\frac{R}{U + B + G + R}

        X = \\frac{\\sqrt{3}}{\\sqrt{2}2}(1 - 2s - m - u)
        Y = \\frac{-1 + 3m + u}{2\\sqrt{2}}
        Z = u - \\frac{1}{4}

    And returned as (X, Y, Z), in that order.

    This is taken from:

        Endler and Mielke, 2005. "Comparing Entire Colour Patterns as Birds
        See Them." *Biological Journal of the Linnean Society*, Vol. 86 No. 4,
        pp. 405-431.
    '''
    def apply(self, x: Dict) -> Dict:
        with _coerce_to_4dim(x):
            if x['image'].shape[3] != 4:
                raise ValueError(
                    f"UBGRtoXYZ operator requires 4 bands, got "
                    f"{x['image'].shape}"
                )

            # First, convert (U, B, G, R) -> (u, s, m, l):
            x['image'] = x['image'] / x['image'].sum(3, keepdims=True)
            u, s, m, l = np.moveaxis(x['image'], 3, 0)  # noqa
            # Second, convert (u, s, m, l) -> (X, Y, Z). Note careful placement
            # of parantheses so that scalars are fully calculated before being
            # multiplied or divided by the array. We assign bands as ZYX.
            x['image'] = np.empty((*x['image'].shape[:3], 3), dtype=np.float32)
            # Since (u + s + m + l = 1, we can rewrite (1 - 2s - m - u) as
            # (l - s), which is more efficient since it requires only one
            # subtraction.
            x['image'][..., 2] = (sqrt(3 / 2) / 2) * (l - s)         # X
            x['image'][..., 1] = (-1 + (3 * m) + u) / (2 * sqrt(2))  # Y
            x['image'][..., 0] = u - 0.25                            # Z

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts


@OPERATOR_REGISTRY.register
class VerticalFlip(Operator):
    '''
    This flips the image vertically.
    '''
    def apply(self, x: Dict) -> Dict:
        x['image'] = x['image'][::-1]
        if x.get('mask') is not None:
            x['mask'] = x['mask'][::-1]
        return x


def load_operator(x: Union[str, Dict]) -> Operator:
    '''
    Loads an operator from a saved JSON file.
    '''
    if isinstance(x, str):
        with open(x, 'r') as op_file:
            x = json.load(op_file)

    op_cls = OPERATOR_REGISTRY.get(x.pop('class'))
    return op_cls(**x)
