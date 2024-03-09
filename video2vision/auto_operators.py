'''
This file contains various operators that change themselves when they are used.
This includes, for example, :class:`AutoAlign`, which uses the first batch of
images it recieves to calculate an alignment, then reuses that alignment for
all of the remaining batches.
'''
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .elementwise import build_linearizer
from .operators import (
    ConcatenateOnBands,
    HoldToken,
    Operator,
    OPERATOR_REGISTRY,
)
from .pipeline import ResetPipeline
from .utils import (
    _coerce_to_4dim,
    _evaluate_ecc_for_warp,
    detect_motion,
    _extract_background,
    extract_samples,
    locate_aruco_markers,
    _prep_for_motion_detection,
)
from .warp import Warp

__all__ = [
    'AlignmentNotFound', 'AutoAlign', 'AutoLinearize', 'AutoOperator',
    'AutoTemporalAlign'
]


class AutoOperator(Operator):
    '''
    An :class:`AutoOperator` is an :class:`Operator` that uses the first batch
    of data it receives to calculate some of its parameters.
    '''


@OPERATOR_REGISTRY.register
class AutoAlign(Warp, AutoOperator):
    '''
    The :class:`AutoAlign` takes in two images and attempts to find a
    homography operation that aligns them, then merges them on the bands. This
    requires a good coarse alignment to work. This supports two methods:
    finding tiepoints using ARUCO markers, and finding alignments using the ECC
    method. The ARUCO method is very fast, but requires that the ARUCO markers
    be present in the scene. The ECC method is very slow, but does not. For
    details on the ECC method, see:

    https://hal.inria.fr/file/index/docid/864385/filename/PAMI_2008_preprint.pdf

    Note that, while the ARUCO method is more robust to the initial alignment
    then the ECC method, it is not able to detect markers that have been
    flipped, and will fail if one image has been flipped compared to the other.
    '''
    num_inputs = 2

    def __init__(self, max_iterations: int = 5000, eps: float = 1e-4,
                 sampling_mode: int = cv2.INTER_LINEAR,
                 num_votes: int = 1, bands: Optional[List[List[int]]] = None,
                 mask: Optional[Tuple[int, int, int, int]] = None,
                 method: str = 'any', coe: Optional[np.ndarray] = None,
                 output_size: Optional[Tuple[int, int]] = None):
        '''
        Args:
            max_iterations (int): Maximum iterations to run the ECC algorithm.
            This is only used by the ECC method.

            eps (float): Tolerance for the change in ECC for terminating the
            algorithm. Be careful not to set this too low or it can keep
            running more or less forever and end up jumping into a bad local
            minimum. This is only used by the ECC method.

            sampling_mode (int): Type of interpolation to use.

            num_votes (int): Run the automatic alignment on this many frames
            and compare the results to pick the best alignment.
            bands (optional, list of list of int): Which bands to use. If not
            set, use all.

            mask (optional, 4-tuple of int): If provided, only use this region
            inside the source image to find the match. Should be provided as
            (left x, top y, right x, bottom y). This is only used by the ECC
            method.

            method (str): Method to use. Choices: 'any' (defaults to ARUCO
            markers if available), 'aruco', 'ecc'.

            coe (optional, :class:`numpy.ndarray`): The warp used in the
            alignment. This is ordinarily calculated from the first batch, but
            we allow it to be passed as an argument so that a fit
            :class:`video2vision.AutoAlign` can be saved to disk and reloaded.

            output_size (optional, pair of int): The size of the output image.
            This is ordinarily calculated from the first batch, but we allow it
            to be passed as an argument so that a fit
            :class:`video2vision.AutoAlign` can be saved to disk and reloaded.
        '''
        self.max_iterations = max_iterations
        self.eps = eps
        self.sampling_mode = sampling_mode
        self.num_votes = num_votes
        self.mask = mask
        if method not in {'any', 'aruco', 'ecc'}:
            raise ValueError(f'Did not recognize method {method}')
        self.method = method
        if coe is None:
            self.coe = self.output_size = None
        else:
            self.coe, self.output_size = np.array(coe), output_size
        self._concatenate = ConcatenateOnBands(bands)

    def apply(self, source: Dict, control: Dict) -> Dict:
        # coe will be None if we haven't found an alignment yet.
        if self.coe is not None:
            source = super().apply(source)
            return self._concatenate(source, control)
        else:
            source = self._find_alignment(source, control)
            return self._concatenate(source, control)

    def _find_alignment(self, source: Dict, control: Dict) -> Dict:
        '''
        This a) finds the alignment between the source and control and b) sets
        the internal properties to correspond to that alignment. It returns the
        warped images. We break this out as a separate method so it can be
        called by some subclasses.
        '''
        if self.method == 'any':
            try:
                return self._find_alignment_by_aruco(source, control)
            except AlignmentNotFound:
                return self._find_alignment_by_ecc(source, control)
        elif self.method == 'aruco':
            return self._find_alignment_by_aruco(source, control)
        elif self.method == 'ecc':
            return self._find_alignment_by_ecc(source, control)
        else:
            raise ValueError(f'Did not recognize method {self.method}')

    def _find_alignment_by_aruco(self, source: Dict, control: Dict) -> Dict:
        '''
        Uses ARUCO markers to find the alignment.
        '''
        if self.output_size is None:
            self.output_size = control['image'].shape[:2][::-1]

        # These will raise RuntimeErrors if no markers are found.
        try:
            source_ts, source_pts, source_ids = locate_aruco_markers(source)
            control_ts, control_pts, control_ids = locate_aruco_markers(
                control
            )
        except RuntimeError:
            raise AlignmentNotFound('Could not locate ARUCO markers')

        # Screen for frames with markers present in both
        joint_ts = sorted(set(source_ts) & set(control_ts))
        if not joint_ts:
            raise AlignmentNotFound('No markers found in common frames')

        size = control['image'].shape[:2][::-1]
        # n tracks how many examples we've compared, so we don't exceed the
        # number of votes
        best_ecc, n = -float('inf'), 0
        # These need to be lists so we can use the index method
        source_ts, control_ts = list(source_ts), list(control_ts)

        for t in joint_ts:
            s_idx, c_idx = source_ts.index(t), control_ts.index(t)
            s_ids, c_ids = list(source_ids[s_idx]), list(control_ids[c_idx])
            if set(s_ids) != set(c_ids):
                continue

            # Find tie points
            src_pts = [source_pts[s_idx][s_ids.index(i)] for i in c_ids]
            src_pts = np.array(src_pts).reshape(-1, 2)
            con_pts = control_pts[c_idx].reshape(-1, 2)

            self.coe = Warp.build_from_tiepoints(src_pts, con_pts, size).coe
            # The copy is necessary because Warp.apply modifies the dictionary
            # in-place and returns it. It does this because it assumes it's
            # been called through __call__, which will have already performed a
            # shallow copy. This is not an expensive op since it's only a
            # shallow copy.
            warped_image = super().apply(copy(source))

            # Test the alignment by calculating ECC on the entire image stack.
            # Skip the test if we're only doing this once.
            if self.num_votes > 1:
                ecc = _evaluate_ecc_for_warp(control, warped_image)
            else:
                ecc = 0

            if ecc > best_ecc:
                best_ecc, best_coe, best_warped = ecc, self.coe, warped_image

            n += 1
            if n >= self.num_votes:
                break

        if best_ecc == -float('inf'):
            raise AlignmentNotFound(
                'AutoAlign failed to find viable alignment'
            )

        self.coe = best_coe

        return best_warped

    def _find_alignment_by_ecc(self, source: Dict, control: Dict) -> Dict:
        '''
        Uses the ECC method to find the alignment.
        '''
        if self.output_size is None:
            self.output_size = control['image'].shape[:2][::-1]

        # We could save some compute by pre-converting everything to grayscale.
        # But that would take up a lot of memory, recomputing the means is
        # cheap, and we're tighter on memory then compute.
        source_image, control_image = source['image'], control['image']
        if 'mask' in control and not control['mask'].all():
            raise NotImplementedError(
                'We do not yet support masks in the control image of an '
                'AutoAlign'
            )

        # Select the frames we'll use for the alignment. We try to keep them
        # roughly evenly spaced to avoid using frames that are too close
        # together in time and therefore too similar to generate different
        # warps.
        if source_image.ndim == 4:
            num_frames = control_image.shape[2]
            num_votes = min(self.num_votes, num_frames)
            ts = np.linspace(0, num_frames - 1, num_votes, dtype=np.int64)
            source_frames = [source_image[:, :, t, :] for t in ts]
            control_frames = [control_image[:, :, t, :] for t in ts]
        else:
            num_frames = num_votes = 1
            source_frames = [source_image]
            control_frames = [control_image]

        warp_matrix = np.eye(3, 3, dtype=np.float32)
        criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
        self.output_size = control_frames[0].shape[:2][::-1]

        # Generate mask for ECC calculations
        if self.mask is not None:
            mask = np.zeros(source_frames[0].shape[:2], dtype=np.uint8)
            mask[self.mask[1]:self.mask[3], self.mask[0]:self.mask[2]] = 1
            if source.get('mask') is not None:
                mask &= source['mask']
        else:
            mask = source.get('mask')

        best_ecc = -float('inf')
        # Iterate through contrl frame, source frame pairs, calculating an
        # alignment for each one.
        for control_frame, source_frame in zip(control_frames, source_frames):
            # This can raise a cv2.error if it fails to converge. In that case,
            # we catch the error and try again with the next pair of frames.
            try:
                _, self.coe = cv2.findTransformECC(
                    # Need to specify the dtype because numpy defaults to
                    # float64 and cv2 needs float32 or uint8.
                    source_frame.mean(2, dtype=np.float32),
                    control_frame.mean(2, dtype=np.float32),
                    # cv2.findTransformECC modifies the warp_matrix in-place,
                    # so we need to clone it to avoid over-writing previous
                    # values.
                    warp_matrix.copy(),
                    cv2.MOTION_HOMOGRAPHY,
                    (criteria, self.max_iterations, self.eps),
                    mask,
                    # By default, the findTransformECC applies a Gaussian blur
                    # to the image prior to running. We can't turn it off, but
                    # we can set the kernel size to 1.
                    1,
                )

            except cv2.error:
                continue

            # The copy is necessary because Warp.apply modifies the dictionary
            # in-place and returns it. It does this because it assumes it's
            # been called through __call__, which will have already performed a
            # shallow copy. This is not an expensive op since it's only a
            # shallow copy.
            warped_image = super().apply(copy(source))

            # Test the alignment by calculating ECC on the entire image stack.
            # Skip the test if we're only doing this once.
            if self.num_votes > 1:
                ecc = _evaluate_ecc_for_warp(control, warped_image)
            else:
                ecc = 0

            if ecc > best_ecc:
                best_ecc, best_coe, best_warped = ecc, self.coe, warped_image

        if best_ecc == -float('inf'):
            raise AlignmentNotFound(
                'AutoAlign failed to find viable alignment'
            )

        self.coe = best_coe

        return best_warped

    def release(self):
        if self.coe is None:
            raise AlignmentNotFound(
                'Pipeline ended before AutoAlign found a viable alignment.'
            )

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'coe': (None if self.coe is None else self.coe.tolist()),
            'output_size': self.output_size,
            'max_iterations': self.max_iterations,
            'eps': self.eps,
            'sampling_mode': self.sampling_mode,
            'num_votes': self.num_votes,
            'mask': self.mask,
            'bands': self._concatenate.bands,
            'method': self.method,
        }


@OPERATOR_REGISTRY.register
class AutoLinearize(AutoOperator):
    '''
    The :class:`AutoLinearize` uses ARUCO markers to localize samples of known
    reflectivity in an image, then calculates a linearization to correct post-
    processing by the camera.

    Note that the ARUCO marker detection procedure is robust to rotation, but
    *not* to flipping, and will fail if the image has been flipped compared to
    the expected orientation.
    '''
    def __init__(self, marker_ids: List[int], marker_points: np.ndarray,
                 sample_points: np.ndarray, expected_values: np.ndarray,
                 method: str = 'poly', order: int = 3, sample_width: int = 25,
                 linearization: Optional[str] = None):
        '''
        Args:
            marker_ids (list of int): IDs of the fiducial markers.

            marker_points (:class:`numpy.ndarray`): Locations of the marker
            corners in a reference image.

            sample_points (:class:`numpy.ndarray`): Locations of the centers of
            the samples in a reference image.

            expected_values (:class:`numpy.ndarray`): Expected pixel magnitudes
            of the samples in a reference image.

            method (str): Method to use to fit the linearization. Choices:
            'poly', 'power'.

            order (int): Order of the polynomial to use for the linearization,
            if using a polynomial.

            sample_width (int): Width of the samples to extract.

            linearization (optional, str): If provided, this is a JSON string
            serializing the linearization operator. This is ordinarily
            calculated from the first batch, but we allow it to be passed as an
            argument so that a fit :class:`video2vision.AutoLinearize` can be
            saved to disk and reloaded.
        '''
        self.marker_ids = np.array(marker_ids)
        self.marker_points = np.array(marker_points)
        self.sample_points = np.array(sample_points)
        self.expected_values = np.array(expected_values)
        self.order = order
        self.sample_width = sample_width
        self.method = method
        if linearization is None:
            self.op = None
        else:
            op_cls = OPERATOR_REGISTRY.get(linearization.pop('class'))
            self.op = op_cls(**linearization)

    def apply(self, x: Dict) -> Union[Dict, HoldToken]:
        if self.op is not None:
            return self.op(x)

        with _coerce_to_4dim(x):
            try:
                sample_points, t = self._locate_samples(x)
            except RuntimeError:
                return HoldToken()

            samples = extract_samples(
                x['image'][:, :, t, :], sample_points, self.sample_width
            )

        self.op = build_linearizer(
            samples,
            self.expected_values,
            method=self.method,
            order=self.order,
        )

        return self.op(x)

    def _locate_samples(self, image: Dict):
        '''
        Locates simple points in an image by locating fiducial marks.

        Args:
            image (Dict): Image to be searched.
        '''
        ts, corners = locate_aruco_markers(image, self.marker_ids)
        t, corners = ts[0], corners[0]

        # Using the corners as tie points, build a warp from the original image
        # to the image we've just received. corners will then be a 3-
        # dimensional np.ndarray in arrangement (Marker ID, Corner, X/Y).
        warp = Warp.build_from_tiepoints(
            self.marker_points.reshape(-1, 2),
            corners.reshape(-1, 2),
            # output_size doesn't matter, so give it a dummy
            (0, 0)
        )

        # Apply that warp to the sample points in the original image to obtain
        # sample points in the new image.
        return warp.apply_points(self.sample_points), t

    def release(self):
        if self.op is None:
            raise RuntimeError(
                'Pipeline ended before AutoLinearize found ARUCO markers.'
            )

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'marker_ids': self.marker_ids.tolist(),
            'marker_points': self.marker_points.tolist(),
            'sample_points': self.sample_points.tolist(),
            'expected_values': self.expected_values.tolist(),
            'method': self.method,
            'order': self.order,
            'sample_width': self.sample_width,
            'linearization': self.op if self.op is None else self.op._to_json()
        }


@OPERATOR_REGISTRY.register
class AutoTemporalAlign(AutoAlign, AutoOperator):
    '''
    The :class:`AutoTemporalAlign` takes in two videos and attempts to align
    them both spatially and temporally. It first looks for a stack of frames
    that have sufficient motion in them to make a valid temporal match. It then
    compares a range of possible temporal shifts to look for the optimal shift.
    For each shift, it uses either the ECC method or the ARUCO method to find
    an optimal spatial alignment to go with the temporal shift. Then, it
    selects the optimal temporal alignment based on the ECC value. This
    requires a good coarse alignment to work.
    '''
    def __init__(self, time_shift_range: Tuple[int, int],
                 max_iterations: int = 5000, eps: float = 1e-4,
                 sampling_mode: int = cv2.INTER_LINEAR,
                 num_votes: int = 1, bands: Optional[List[List[int]]] = None,
                 mask: Optional[Tuple[int, int, int, int]] = None,
                 method: str = 'any', motion_area_threshold: float = 0.005,
                 motion_difference_threshold: float = 0.1,
                 coe: Optional[np.ndarray] = None,
                 output_size: Optional[Tuple[int, int]] = None,
                 time_shift: Optional[int] = None):
        '''
        Args:
            time_shift_range (tuple of int): Range of temporal shifts to
            evaluate.

            max_iterations (int): Maximum iterations to run the ECC algorithm.
            This is only used by the ECC method.

            eps (float): Tolerance for the change in ECC for terminating the
            algorithm. Be careful not to set this too low or it can keep
            running more or less forever and end up jumping into a bad local
            minimum. This is only used by the ECC method.

            sampling_mode (int): Type of interpolation to use.

            num_votes (int): Run the automatic alignment on this many frames
            and compare the results to pick the best alignment.
            bands (optional, list of list of int): Which bands to use. If not
            set, use all.

            mask (optional, 4-tuple of int): If provided, only use this region
            inside the source image to find the match. Should be provided as
            (left x, top y, right x, bottom y). This is only used by the ECC
            method.

            method (str): Method to use. Choices: 'any' (defaults to ARUCO
            markers if available), 'aruco', 'ecc'.

            motion_area_threshold (float): The area_threshold value to use in
            the :func:`detect_motion` function.

            motion_difference_threshold (float): The difference_threshold value
            to use in the :func:`detect_motion` function.

            coe (optional, :class:`numpy.ndarray`): The warp used in the
            alignment. This is ordinarily calculated from the first batch, but
            we allow it to be passed as an argument so that a fit
            :class:`video2vision.AutoTemporalAlign` can be saved to disk and
            reloaded.

            output_size (optional, pair of int): The size of the output image.
            This is ordinarily calculated from the first batch, but we allow it
            to be passed as an argument so that a fit
            :class:`video2vision.AutoTemporalAlign` can be saved to disk and
            reloaded.

            time_shift (optional, int): The amount of the time shift. This is
            ordinarily calculated from the first batch, but we allow it to be
            passed as an argument so that a fit
            :class:`video2vision.AutoTemporalAlign` can be saved to disk and
            reloaded.
        '''
        super().__init__(
            max_iterations=max_iterations, eps=eps,
            sampling_mode=sampling_mode, mask=mask, num_votes=num_votes,
            bands=bands, method=method, coe=coe, output_size=output_size,
        )
        self.time_shift_range = time_shift_range
        self.time_shift = time_shift
        self.motion_area_threshold = motion_area_threshold
        self.motion_difference_threshold = motion_difference_threshold
        self.buff = None
        self.buff_names = None
        self.source_background = None
        self.control_background = None
        # This is used to track whether a batch has been skipped, and therefore
        # whether the operator needs to raise a ResetPipeline exception when it
        # finds the correct alignment.
        self.skipped_batch = False
        # This is used to track whether any batch has contained enough frames
        # to POTENTIALLY be used for alignment, even if it couldn't be because
        # it didn't have enough motion. This is used to raise a more explicit
        # error if the batch size is set too small, rather than just saying we
        # couldn't find an alignment.
        self.seen_enough_frames = False

    def apply(self, source: Dict, control: Dict) -> Union[Dict, HoldToken]:
        if self.coe is not None:
            source, control = self._shift(source, control, self.time_shift)
            # This is in place to handle when UV, VIS videos have different
            # length.
            with _coerce_to_4dim(source), _coerce_to_4dim(control):
                nf = min(source['image'].shape[2], control['image'].shape[2])
                source['image'] = source['image'][:, :, :nf, :]
                control['image'] = control['image'][:, :, :nf, :]

            return super().apply(source, control)

        # Check for motion prior to actually running the alignment
        if self.source_background is None:
            self.source_background = _prep_for_motion_detection(
                _extract_background(source)
            )
        if self.control_background is None:
            self.control_background = _prep_for_motion_detection(
                _extract_background(control)
            )

        source_motion = self._detect_motion(source, self.source_background)
        control_motion = self._detect_motion(control, self.control_background)

        # Note the need to trim in case source, control have different number
        # of frames.
        with _coerce_to_4dim(source), _coerce_to_4dim(control):
            nf = min(source['image'].shape[2], control['image'].shape[2])
        source_motion, control_motion = source_motion[:nf], control_motion[:nf]
        min_batch_size = max(abs(t) for t in self.time_shift_range)
        self.seen_enough_frames |= (nf >= min_batch_size)

        # We need to ensure that, under all possible time shifts, at least some
        # frames with motion will be available to compare. If they do not, we
        # return a HoldToken, which aborts the graph downstream of this
        # operation.
        for shift in range(*self.time_shift_range):
            if shift == 0:
                is_moving = source_motion & control_motion
            elif shift > 0:
                is_moving = source_motion[:-shift] & control_motion[shift:]
            elif shift < 0:
                is_moving = source_motion[-shift:] & control_motion[:shift]
            if not is_moving.any():
                self.skipped_batch = True
                return HoldToken()

        # If we go this far, then we have adequate motion and may proceed.
        # Let's also discard those backgrounds to save memory while we're at
        # it.
        self.source_background = self.control_background = None
        out = self._find_alignment(source, control)

        # If we've skipped a batch, then reset the pipeline and process
        # everything over again.
        if self.skipped_batch:
            raise ResetPipeline()

        return out

    def _detect_motion(self, images, background):
        '''
        This is a convenience method wrapping the :func:`detect_motion`
        function.
        '''
        return detect_motion(
            images,
            area_threshold=self.motion_area_threshold,
            difference_threshold=self.motion_difference_threshold,
            background=background,
        )

    def _find_alignment(self, source: Dict, control: Dict) -> Dict:
        '''
        This finds the spatial and temporal alignment between the source and
        controls, and returns the aligned outputs. This assumes that adequate
        motion is present in both.
        '''
        best_ecc = -float('inf')

        # Get the number of frames available.
        with _coerce_to_4dim(source), _coerce_to_4dim(control):
            t = min(source['image'].shape[2], control['image'].shape[2])

        # Iterate through possible time shifts
        for time_shift in range(*self.time_shift_range):
            # If the time shift exceeds the number of frames available, then
            # skip.
            if abs(time_shift) >= t:
                continue

            shifted_source, shifted_control = self._shift(
                source, control, time_shift, no_buffer=True,
            )

            # This can raise an AlignmentNotFound error if we fail to find an
            # alignment.
            try:
                warped_source = super()._find_alignment(
                    shifted_source, shifted_control
                )
            except AlignmentNotFound:
                continue

            ecc = _evaluate_ecc_for_warp(shifted_control, warped_source)
            if ecc > best_ecc:
                best_ecc, best_shift, best_coe = ecc, time_shift, self.coe

        if best_ecc == -float('inf'):
            raise AlignmentNotFound(
                'AutoTemporalAlign failed to find viable alignment'
            )

        self.coe, self.time_shift = best_coe, best_shift

        return self.apply(source, control)

    def release(self):
        if self.coe is None:
            if not self.seen_enough_frames:
                min_batch_size = max(abs(t) for t in self.time_shift_range)
                raise AlignmentNotFound(
                    f'Pipeline ended without seeing enough frames to find a '
                    f'temporal alignment. Check that batch_size of the loaders'
                    f' is greater than {min_batch_size}.'
                )
            else:
                raise AlignmentNotFound(
                    'Pipeline ended before AutoTemporalAlign found a viable '
                    'alignment.'
                )

    def reset(self):
        self.buff = None
        self.buff_names = None
        self.source_background = None
        self.control_background = None
        self.skipped_batch = False
        return super().reset()

    def set_batch_size(self, batch_size: int):
        if self.coe is None:
            min_batch_size = max(abs(t) for t in self.time_shift_range)
            if batch_size <= min_batch_size:
                raise ValueError(
                    f'Batch size ({batch_size}) too small for time shift range'
                    f'({self.time_shift_range}); needs at least '
                    f'{min_batch_size}'
                )

    def _shift(self, source: Dict, control: Dict, shift: int = 0,
               no_buffer: bool = False):
        '''
        Shifts the source and control. If positive, shifts source forward. If
        negative, shifts control backward. Unlike
        :class:`video2vision.TemporalShift`, this trims the video instead of
        padding it.

        Args:
            source (dict): Source image.

            control (dict): Control image.

            shift (int): Number of frames to shift.

            no_buffer (bool): If set, do not use the buffer to store the
            trimmed frames.
        '''
        source, control = copy(source), copy(control)

        if shift == 0:
            return source, control

        shifted, static = (source, control) if shift > 0 else (control, source)
        shift = abs(shift)

        with _coerce_to_4dim(shifted), _coerce_to_4dim(static):
            image, buff = np.split(shifted['image'], [-shift], axis=2)

            if no_buffer or self.buff is None:
                _, static['image'] = np.split(static['image'], [shift], axis=2)

            if not no_buffer:
                if self.buff is not None:
                    image = np.concatenate((self.buff, image), axis=2)
                self.buff = buff

            shifted['image'] = image

        if 'names' in static and (no_buffer or self.buff_names is None):
            static['names'] = static['names'][shift:]

        if 'names' in shifted:
            buff = shifted['names'][-shift:]
            shifted['names'] = shifted['names'][:-shift]
            if not no_buffer:
                if self.buff_names is not None:
                    shifted['names'] = self.buff_names + shifted['names']
                self.buff_names = buff

        return source, control

    def _to_json(self) -> Dict:
        rtn = super()._to_json()
        rtn.update({
            'motion_area_threshold': self.motion_area_threshold,
            'motion_difference_threshold': self.motion_difference_threshold,
            'time_shift_range': self.time_shift_range,
            'time_shift': self.time_shift,
        })
        return rtn


class AlignmentNotFound(Exception):
    '''
    Represents an error where an automatic operator was expected to align two
    sets of frames, but was unable to.
    '''
