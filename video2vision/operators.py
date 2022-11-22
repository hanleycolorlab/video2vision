'''
This provides various :class:`Operator` s, which represent operations performed
on an image. We build them as distinct classes so that they can be saved to
keep track of what operations were performed to produce an image, and reused on
other images.
'''
from copy import copy
import json
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .utils import _coerce_to_3dim, _coerce_to_dict, Registry

__all__ = [
    'Operator', 'ConcatenateOnBands', 'HorizontalFlip', 'LinearMap',
    'load_operator', 'OPERATOR_REGISTRY', 'Pad', 'Resize', 'Rotate',
    'TemporalShift', 'Translation', 'VerticalFlip', 'Warp'
]

# This provides a registry of operators. This is used when saving and restoring
# Operators, since it allows us to map from operator name to the operator
# class.
OPERATOR_REGISTRY = Registry()


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

    def __call__(self, *xs) -> Dict:
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
class TemporalShift(Operator):
    '''
    Shifts a video forward in time.
    '''
    def __init__(self, shift: int):
        '''
        Args:
            shift (int): The number of frames to shift. If positive, shifts the
            video forward; if negative, shifts it backward.
        '''
        if shift < 0:
            raise ValueError('Shift must be positive')
        self.shift = shift
        self.buff = None

    def apply(self, x: Dict) -> Dict:
        if self.shift == 0:
            return x

        if x['image'].ndim != 4:
            raise ValueError(
                'TemporalShift cannot be applied to single images'
            )

        w, h, t, c = x['image'].shape

        if t <= self.shift:
            raise ValueError(
                'Temporal shift requires shift to be smaller than batch size'
            )

        # We do a copy because the slice returns a view, keeping a hidden
        # reference to the original image array which will prevent it from
        # being garbage-collected.
        buff = self.buff
        self.buff = x['image'][:, :, -self.shift:, :].copy()
        if buff is not None:
            x['image'] = np.concatenate(
                (buff, x['image'][:, :, :-self.shift, :]), axis=2
            )
        else:
            pad = ((0, 0), (0, 0), (self.shift, 0), (0, 0))
            x['image'] = np.pad(x['image'][:, :, :-self.shift], pad)

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        return pts

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'shift': self.shift,
        }


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


@OPERATOR_REGISTRY.register
class Warp(Operator):
    '''
    This class wraps the :func:`cv2.warpAffine` and :func:`cv2.warpPerspective`
    functions.
    '''
    def __init__(self, coe: np.ndarray, output_size: Tuple[int, int],
                 sampling_mode: int = cv2.INTER_LINEAR):
        '''
        Args:
            coe (:class:`np.ndarray`): Transform matrix.

            output_size (tuple of int): The height and width of the output
            image.

            sampling_mode (int): Interpolation mode.
        '''
        self.coe = np.array(coe).astype(np.float32)
        if self.coe.shape not in {(2, 3), (3, 3)}:
            raise ValueError(
                f'Transform matrix shape not valid: {self.coe.shape}'
            )
        self.output_size = output_size
        self.sampling_mode = sampling_mode

    def apply(self, x: Dict) -> Dict:
        warp = (cv2.warpAffine if self.is_affine else cv2.warpPerspective)

        # If this is a video, we convert it to (H, W, C * T) arrangement. cv2
        # will treat this as an image with an unusually large number of
        # channels and be perfectly fine with it.
        with _coerce_to_3dim(x):
            if x.get('mask') is None:
                x['mask'] = np.ones(x['image'].shape[:2], dtype=np.uint8)
            # Note: output_size needs to be a tuple; some versions of OpenCV
            # will raise a SystemError if it's not.
            x['image'] = warp(
                x['image'], self.coe, tuple(self.output_size),
                flags=self.sampling_mode,
            )
            x['mask'] = warp(
                x['mask'], self.coe, tuple(self.output_size),
                flags=cv2.INTER_NEAREST,
            )

        return x

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        # For unknown reasons, cv2.perspectiveTransform requires the points to
        # be in (H, W, X/Y) arrangement. They also require float32 or 64.
        pts = pts.astype(np.float32).reshape(-1, 1, 2)
        if self.coe.shape[0] == 2:
            pts = cv2.transform(pts, self.coe)
        else:
            pts = cv2.perspectiveTransform(pts, self.coe)
        return pts.reshape(-1, 2)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'coe': self.coe.tolist(),
            'output_size': self.output_size,
            'sampling_mode': self.sampling_mode,
        }

    @classmethod
    def build_from_tiepoints(cls, source_points: np.ndarray,
                             control_points: np.ndarray,
                             control_size: Tuple[int, int],
                             method: int = cv2.RANSAC) -> Operator:
        '''
        Builds a :class:`Warp` operator from a set of tiepoints between the
        image to be warped and a target image.

        Args:
            source_points (:class:`np.ndarray`): Tie points in the image to be
            warped.
            control_points (:class:`np.ndarray`): Points to warp the tie points
            to.
            control_size (tuple of int): Size of the output image.
            method (int): Method to use in warping.
        '''
        source_points = np.array(source_points)
        control_points = np.array(control_points)
        coe, _ = cv2.findHomography(source_points, control_points, method)
        return Warp(coe, control_size)

    @property
    def is_affine(self) -> bool:
        return self.coe.shape == (2, 3)


@OPERATOR_REGISTRY.register
class Rotate(Warp):
    '''
    This class performs a rotation.
    '''
    def __init__(self, angle: float, output_size: Tuple[int, int],
                 center: Optional[Tuple[int, int]] = None):
        '''
        Args:
            angle (float): Angle of the rotation.
            output_size (tuple of int): The height and width of the output
            image.
            center (optional, tuple of int): The center point in the input
            image to rotate around.
        '''
        # Note: copied from imutils to ensure consistency with older versions.
        if center is None:
            center = (output_size[1] // 2, output_size[0] // 2)
        self.angle, self.center = angle, center
        # Note: center needs to be a tuple; some versions of OpenCV will raise
        # a SystemError if it's not.
        coe = cv2.getRotationMatrix2D(tuple(center), angle, scale=1)
        super().__init__(coe=coe, output_size=output_size)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'angle': self.angle,
            'output_size': self.output_size,
            'center': self.center,
        }


@OPERATOR_REGISTRY.register
class Translation(Warp):
    '''
    This class performs a translation.
    '''
    def __init__(self, shift_x: float, shift_y: float,
                 output_size: Tuple[int, int]):
        '''
        Args:
            shift_x (float): Shift in x coordinate.
            shift_y (float): Shift in y coordinate.
            output_size (tuple of int): The height and width of the output
            image.
        '''
        self.shift_x, self.shift_y = shift_x, shift_y
        coe = np.array([[1., 0., shift_x], [0., 1., shift_y]])
        super().__init__(coe=coe, output_size=output_size)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'shift_x': self.shift_x,
            'shift_y': self.shift_y,
            'output_size': self.output_size,
        }


def load_operator(x: Union[str, Dict]) -> Operator:
    '''
    Loads an operator from a saved JSON file.
    '''
    if isinstance(x, str):
        with open(x, 'r') as op_file:
            x = json.load(op_file)

    op_cls = OPERATOR_REGISTRY.get(x.pop('class'))
    return op_cls(**x)
