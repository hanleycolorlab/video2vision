'''
This includes the various operators implementing homography warp operations.
'''
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .operators import Operator, OPERATOR_REGISTRY
from .utils import _coerce_to_3dim

__all__ = ['Rotate', 'Translation', 'Warp']


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
