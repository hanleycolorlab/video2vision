from contextlib import contextmanager
from glob import glob
import json
import os
from statistics import mean
import subprocess
import tempfile
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import tifffile
    has_tiff = True
except ImportError:
    has_tiff = False

try:
    import rawpy
    has_rawpy = True
except ImportError:
    has_rawpy = False

from .operators import Operator, OPERATOR_REGISTRY
from .utils import _coerce_to_image

__all__ = [
    'load', 'Loader', 'MisshapenImageError', 'OutOfInputs', 'save',
    'trim_video', 'Writer',
]

# Default video frame rate (used by FFmpeg)
_VIDEO_FPS = 23.976023976023978

# This is a global variable used to track whether the Loader/Writer should
# read/write from disk or to an internal buffer. This internal buffer is used
# to apply a Pipeline to images in memory instead of running the normal way.
_READ_WRITE_FROM_TO_BUFFER = False

_SUPPORTED_EXTENSIONS = [
    'arw', 'jpeg', 'jpg', 'mov', 'mp4', 'nef', 'png', 'raw', 'tif', 'tiff'
]


def load(path: str, out: Optional[np.ndarray] = None,
         for_display: bool = False) -> np.ndarray:
    '''
    Convenience function for loading images from disk. Dispatches to
    appropriate backend. Images will be converted to 32-bit floating point
    numbers scaled to [0, 1] before being returned.

    Args:
        path (str): Path to image to load.
        out (optional, :class:`numpy.ndarray`): If provided, this is the buffer
            to load the image into.
        for_display (bool): This is used to indicate that the image is needed
            for display, rather than for analysis. The image will be returned
            scaled [0, 255] instead of [0, 1]. In addition, if the image's
            original dtype is uint8, it will be left as uint8, whereas
            otherwise it will be coerced to float32.
    '''
    # Some of the function calls below don't raise an error if the file doesn't
    # exist, they just fail silently. So let's check explicitly.

    # TODO: Make out argument work with uint8
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(('.tif', '.tiff')):
        if not has_tiff:
            raise ImportError('tifffile is needed to read tif files')
        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes.upper()
            image = tif.asarray()
        if axes in {'CYX', 'QYX', 'SXY'}:
            image = np.moveaxis(image, 0, -1)
        elif axes == 'YX':
            image = image.reshape(*image.shape, 1)
        elif axes not in {'YXC', 'YXQ', 'YXS'}:
            raise ValueError(
                f'Could not interpret TIFF metadata; axes are {axes}'
            )

        # Rescale to [0, 1] and float32
        if not for_display:
            image = _convert_and_scale_uint8(image, out=out)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

    elif path.lower().endswith('.mp4'):
        reader = cv2.VideoCapture(path)
        frames, ret = [], True
        while ret and reader.isOpened():
            ret, frame = reader.read()
            if ret:
                frames.append(frame)
        reader.release()
        image = np.stack(frames, axis=2)
        # Rescale to [0, 1] and float32
        if not for_display:
            image = _convert_and_scale_uint8(image, out=out)

    elif path.lower().endswith(('.arw', '.nef')):
        if not has_rawpy:
            raise ImportError('rawpy is needed to read arw files')

        if for_display:
            with rawpy.imread(path) as raw_file:
                image = raw_file.postprocess()

        else:
            with rawpy.imread(path) as raw_file:
                image = raw_file.postprocess(
                    # Prevents gamma correction
                    gamma=(1, 1),
                    output_color=rawpy.ColorSpace.raw,
                    no_auto_scale=True,
                    no_auto_bright=True,
                    # Output will be maximum precision
                    output_bps=16,
                )
                white_level = np.array([
                    raw_file.camera_white_level_per_channel[0],
                    mean((raw_file.camera_white_level_per_channel[1],
                          raw_file.camera_white_level_per_channel[3])),
                    raw_file.camera_white_level_per_channel[2],
                ])

            # Rescales to [0, 1] and float32
            image = image.astype(np.float32)
            image = np.divide(
                image,
                white_level.reshape(1, 1, 3),
                None if (out is None) else out[:, :, ::-1],
            )

        # Reverse channels from RGB to BGR
        image = image[:, :, ::-1]

    else:
        # This will return None if it can't read the path
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f'Image {path} could not be read')
        # Rescale to [0, 1] and float32
        if not for_display:
            image = _convert_and_scale_uint8(image, out=out)

    return _coerce_to_image(image)


def save(image: np.ndarray, path: str):
    '''
    Convenience function for saving images to disk. Dispatches to appropriate
    backend.

    Args:
        image (:class:`np.ndarray`): Image to save.

        path (str): Path to save image to.
    '''
    # Rescale from [0, 1] -> [0, 256]
    image = 256 * image

    if image.ndim == 4:
        # Use Writer class for video output instead
        raise NotImplementedError(
            'Video writing not supported in save() function. '
            'Use Writer class instead.'
        )

    elif path.lower().endswith(('.tif', '.tiff')):
        if not has_tiff:
            raise ImportError('tifffile is needed to write tif files')
        tifffile.imwrite(
            path,
            np.moveaxis(image, -1, 0),
            imagej=True,
            # This specifies metadata to be included that specifies the 0th
            # dimension is the channel dimension. This eliminates an issue
            # where viewer program's interpretation of the image layout is
            # inconsistent.
            metadata={'axes': 'CYX'},
        )

    elif path.lower().endswith(('.nef', '.arw')):
        raise NotImplementedError('Saving in raw form is not supported')

    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.imwrite(path, image)


def _detect_video_properties(path: str) -> tuple:
    '''
    Detect bit depth and frame rate of a video file using ffprobe.

    Args:
        path (str): Path to video file

    Returns:
        tuple: (bit_depth, fps) - defaults to (8, 23.976) if detection fails
    '''
    bit_depth = 8
    fps = 23.976023976023978

    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=pix_fmt,r_frame_rate',
            '-of', 'json',
            path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data['streams'][0]

            # Detect bit depth from pixel format
            pix_fmt = stream.get('pix_fmt', '')
            if '10' in pix_fmt or 'p010' in pix_fmt:
                bit_depth = 10
            elif '12' in pix_fmt:
                bit_depth = 12
            elif '16' in pix_fmt:
                bit_depth = 16

            # Detect frame rate
            r_frame_rate = stream.get('r_frame_rate', '')
            if r_frame_rate and '/' in r_frame_rate:
                num, denom = r_frame_rate.split('/')
                fps = float(num) / float(denom)
    except Exception:
        pass

    return bit_depth, fps


@OPERATOR_REGISTRY.register
class Loader(Operator):
    '''
    This is an operator that loads images from disk, acting as a root of the
    :class:`Pipeline` DAG. It can be used by:

    .. code-block:: python

        loader = Loader(path)
        image = loader()

    It will raise a :class:`OutOfInputs` exception when it runs out of images
    to load. It returns images in (H, W, C) arrangement if the batch size is 1,
    and returns a stack of images in (H, W, T, C) arrangement if the batch size
    is larger than 1.
    '''
    num_inputs = 0

    def __init__(self, path: Optional[Union[str, Iterator[str]]],
                 expected_size: Tuple[int, int],
                 batch_size: int = 1, num_channels: int = 3):
        '''
        Args:
            path (str or iterator of str): Path(s) to the images to load. This
            can be a directory, in which case everything in the directory will
            be loaded, or it can be a path to a single image, or it can be a
            string with wildcards.
            expected_size (pair of int): The inputs must be of this size. This
            should be (width, height).
            batch_size (int): Number of images to return at a time.
            num_channels (int): Number of channels to expect in inputs.
        '''
        self.set_path(path)
        self.batch_size = batch_size
        self.expected_size = expected_size
        self.num_channels = num_channels
        self.bit_depth = 8  # Default to 8-bit
        self.fps = _VIDEO_FPS  # Default FPS
        # This is used to provide external inputs from memory to a pipeline,
        # instead of loading from disk. It is only consulted if the global
        # variable _READ_WRITE_FROM_BUFFER is true.
        self.buff: Optional[np.ndarray] = None

        # Auto-detect bit depth and FPS for video files
        if path and isinstance(path, str):
            if path.lower().endswith(('.mp4', '.mov')):
                self.bit_depth, self.fps = _detect_video_properties(path)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[str]]]:
        # If _READ_WRITE_FROM_BUFFER, we are providing inputs from memory, not
        # loading from disk. This is used to apply Pipelines to images in
        # memory.
        if _READ_WRITE_FROM_TO_BUFFER:
            for image in self.buff:
                self._check_size(image)
                yield image, None

        else:
            # We arrange the buffer in order THWC instead of the usual order
            # HWTC, because this reduces the time required to copy frames in by
            # a factor of x6. We then move the axis prior to returning to
            # convert it to HWTC. This *does* mean that the returned array is
            # not contiguous, but operators are not supposed to assume that it
            # will be.
            self.buff = np.empty(
                (self.batch_size, *self.expected_size[::-1],
                 self.num_channels),
                dtype=np.float32
            )
            # self.t keeps track of the next entry in the buffer to fill.
            self.t = 0

            for path, reader in zip(self.paths, self._readers):
                name, _ = os.path.splitext(os.path.basename(path))
                if reader is not None:
                    ret, frame = reader.read()
                    while ret and reader.isOpened():
                        self._check_size(frame)
                        # Rescale to [0, 1] before returning
                        frame = _convert_and_scale_uint8(
                            frame, out=self.buff[self.t % self.batch_size]
                        )
                        self.t += 1
                        yield frame, name
                        ret, frame = reader.read()
                else:
                    # load handles rescaling for us
                    image = load(path, out=self.buff[self.t % self.batch_size])
                    self.t += 1
                    self._check_size(image)
                    yield image, name

    def __len__(self) -> int:
        if _READ_WRITE_FROM_TO_BUFFER:
            return len(self.buff)
        else:
            return sum(int(_get_num_frames(r)) for r in self._readers)

    def _check_size(self, image: np.ndarray):
        if image.shape != (*self.expected_size[::-1], self.num_channels):
            raise MisshapenImageError(
                f'Image does not match expected size: {image.shape} vs '
                f'{(*self.expected_size[::-1], self.num_channels)}'
            )

    def apply(self) -> Dict:
        # If _READ_WRITE_FROM_BUFFER, we are providing inputs from memory, not
        # loading from disk. This is used to apply Pipelines to images in
        # memory.
        if _READ_WRITE_FROM_TO_BUFFER:
            if self.buff is not None:
                out, self.buff = np.moveaxis(self.buff, 0, 2), None
                return {'image': out, 'final': True}
            else:
                raise OutOfInputs('Buffer is empty')

        if len(self) == 0:
            raise FileNotFoundError(self.original_path)

        # We arrange the buffer in order THWC instead of the usual order HWTC,
        # because this reduces the time required to copy frames in by a factor
        # of x6. We then move the axis prior to returning to convert it to
        # HWTC. This *does* mean that the returned array is not contiguous, but
        # operators are not supposed to assume that it will be.
        self.buff = np.empty(
            (self.batch_size, *self.expected_size[::-1], self.num_channels),
            dtype=np.float32
        )
        names = []

        try:
            for _ in range(self.batch_size):
                _, name = next(self._data_iter)
                names.append(name)
        except StopIteration:
            if len(names) == 0:
                raise OutOfInputs()

        n = self.t % self.batch_size
        self.buff = self.buff[:(n if (n > 0) else self.batch_size), :, :, :]

        # final tracks whether this is the last batch to process
        result = {
            'image': np.moveaxis(self.buff, 0, 2),
            'names': names,
            'final': (self.t >= len(self)),
            'bit_depth': self.bit_depth,  # Always include bit depth
            'fps': self.fps,  # Always include FPS
        }

        return result

    def get_frame(self, t: int, for_display: bool = False) -> np.ndarray:
        '''
        Retrieves a single frame and returns it.

        Args:
            t (int): Index of the frame to retrieve.
            for_display (bool): This is used to indicate that the image is
                needed for display, rather than for analysis. The image will be
                returned scaled [0, 255] instead of [0, 1]. In addition, if the
                image's original dtype is uint8, it will be left as uint8,
                whereas otherwise it will be coerced to float32.
        '''
        if _READ_WRITE_FROM_TO_BUFFER:
            return self.buff[t]

        n_frames = 0
        for path, reader in zip(self.paths, self._readers):
            n_frames += _get_num_frames(reader)
            if n_frames > t:
                break
        else:
            if n_frames == 0:
                raise FileNotFoundError(self.original_path)
            else:
                raise ValueError(f'{t} out of range: {len(self)}')

        if reader is None:
            # load handles rescaling for us
            image = load(path, for_display=for_display)
        else:
            reader.set(cv2.CAP_PROP_POS_FRAMES, t)
            _, image = reader.read()
            if not for_display:
                image = _convert_and_scale_uint8(image)

        self._check_size(image)

        return image

    def reset(self):
        '''
        Resets the :class:`Loader` to the start of its inputs.
        '''
        self._data_iter = iter(self)
        self.buff, self.t = None, 0
        for reader in self._readers:
            if reader is not None:
                reader.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_batch_size(self, batch_size: int):
        '''
        Sets the batch size of frames drawn per iteration.
        '''
        self.batch_size = batch_size

    def set_path(self, paths: Optional[Union[str, Iterator[str]]]):
        '''
        Sets the input path.
        '''
        if paths is None:
            self.original_path, self.paths, self._readers = None, None, []
        else:
            # We use original_paths to hold the value passed by the user, for
            # error-reporting purposes.
            self.original_path = paths
            self.paths, self._readers = [], []
            if isinstance(paths, str):
                paths = [paths]

            for path in paths:
                if os.path.isdir(path):
                    path = os.path.join(path, '*')
                self.paths += sorted(glob(path))

            for path in self.paths:
                if path.lower().endswith('.mp4'):
                    self._readers.append(cv2.VideoCapture(path))
                else:
                    self._readers.append(None)

        self.reset()

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'batch_size': self.batch_size,
            'expected_size': self.expected_size,
            'num_channels': self.num_channels,
            'path': self.paths
        }


@OPERATOR_REGISTRY.register
class Writer(Operator):
    '''
    This is an :class:`Operator` that writes finished images to disk. It acts
    as a leaf of the :class:`Pipeline` DAG.
    '''

    def __init__(self, path: Optional[str] = None,
                 extension: Optional[str] = None,
                 separate_bands: bool = False, suffix: str = '',
                 bit_depth: Optional[int] = None,
                 codec: str = 'auto',
                 audio_source: Optional[str] = None,
                 include_audio: bool = True):
        '''
        Args:
            path (str): Path to the directory to write images to.
            extension (optional, str): Extension to use in writing images. If
            not set, attempts to infer from path; if unsuccessful, assumes
            'tif'.

            separate_bands (bool): Whether to save each band as a separate
            output file, instead of a single output file.

            suffix (str): Append this to the end of output file names. This is
            only used if the output path is a directory, not an MP4.

            bit_depth (optional, int): Output bit depth (8, 10, 12, or 16).
            If None, uses 8-bit. For >8-bit, uses FFmpeg subprocess.

            codec (str): Video codec for high bit depth output.
            Options: 'auto', 'prores', 'hevc', 'h264'.
            'auto' selects prores for .mov, hevc for .mp4.

            audio_source (optional, str): Path to video file to extract audio
            from. If None and include_audio is True, no audio will be added.
            Typically this should be the VIS camera video path.

            include_audio (bool): Whether to include audio in video output.
            Defaults to True. Only applies to video formats (mp4, mov).
        '''
        if extension is None:
            # Check for extension in path
            _, extension = os.path.splitext(path or '')
            extension = extension[1:].lower()
            if extension not in _SUPPORTED_EXTENSIONS:
                extension = 'tif'
        if extension.lower() in {'arw', 'raw'}:
            raise NotImplementedError('Writer does not support RAW format')
        self.extension = extension
        # This will be used to hold cv2.VideoWriter if present
        self._writer = None
        self.separate_bands = separate_bands
        self.set_path(path)
        self.suffix = suffix
        # This is used to return inputs to memory from a pipeline, instead of
        # writing to disk. It is only used if the global variable
        # _READ_WRITE_FROM_BUFFER is true.
        self.buff = []

        # High bit depth support
        self.bit_depth = bit_depth or 8
        self.codec = codec
        self.fps = None  # Will be set from input data
        self._ffmpeg_process = None
        self._ffmpeg_stderr = None
        self._frame_count = 0

        # Audio support
        self.audio_source = audio_source
        self.include_audio = include_audio
        self._audio_temp_file = None

    def apply(self, x: Dict):
        image = x['image']
        num_frames = 1 if (image.ndim == 3) else image.shape[2]
        names = x.get('names', [None] * num_frames)

        # Capture FPS from input data if not already set
        if self.fps is None and 'fps' in x:
            self.fps = x['fps']

        if num_frames != len(names):
            raise RuntimeError(
                f'Mismatch between number of frames {num_frames} and number of'
                f' names {names}'
            )

        if image.ndim == 3:
            self._write(image, names[0])
        else:
            for t in range(image.shape[2]):
                self._write(image[:, :, t, :], names[t])

    def __del__(self):
        self.release()

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'extension': self.extension,
            'separate_bands': self.separate_bands,
        }

    def _setup_ffmpeg_writer(self, width: int, height: int, path: str):
        '''Setup FFmpeg subprocess for video writing with optional audio.'''
        # Determine codec
        if self.codec == 'auto':
            if self.extension == 'mov':
                self.codec = 'prores'
            else:
                self.codec = 'hevc'

        # Use detected FPS or fallback to default
        fps = self.fps if self.fps is not None else _VIDEO_FPS

        print(f"Using {self.bit_depth}-bit {self.codec} encoder")
        print(f"  Output: {path}")
        print(f"  Dimensions: {width}x{height}")
        print(f"  FPS: {fps:.3f}")

        # Check if audio source exists
        has_audio = (
            self.include_audio and
            self.audio_source is not None and
            os.path.exists(self.audio_source)
        )

        if has_audio:
            print(f"  Audio: {os.path.basename(self.audio_source)}")

        # Configure codec-specific parameters
        if self.codec == 'prores':
            pix_fmt = 'yuv422p10le' if self.bit_depth >= 10 else 'yuv422p'
            codec_params = ['-c:v', 'prores_ks', '-profile:v', '3']
        elif self.codec == 'hevc':
            pix_fmt = 'yuv420p10le' if self.bit_depth >= 10 else 'yuv420p'
            codec_params = [
                '-c:v', 'libx265', '-crf', '18',
                '-preset', 'slow',
                '-x265-params', 'profile=main10',
            ]
        else:  # h264
            pix_fmt = 'yuv420p'
            codec_params = ['-c:v', 'libx264', '-crf', '18']

        # FFmpeg expects BGR format from OpenCV/numpy
        input_pix_fmt = 'bgr48le' if self.bit_depth > 8 else 'bgr24'

        # Build FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', input_pix_fmt,
            '-r', str(fps),
            '-i', '-',  # Video from stdin
        ]

        # Add audio input if available
        if has_audio:
            ffmpeg_cmd.extend(['-i', self.audio_source])

        # Add video codec parameters
        ffmpeg_cmd.extend(codec_params)
        ffmpeg_cmd.extend(['-pix_fmt', pix_fmt])

        # Add audio codec parameters
        if has_audio:
            ffmpeg_cmd.extend([
                '-c:a', 'aac',      # Encode audio as AAC
                '-map', '0:v:0',    # Map video from first input (stdin)
                '-map', '1:a:0?',   # Map audio from second input (optional)
                '-shortest'         # Match shortest stream duration
            ])
        else:
            ffmpeg_cmd.append('-an')  # No audio

        # Output path
        ffmpeg_cmd.append(path)

        # Start FFmpeg process
        self._ffmpeg_stderr = tempfile.NamedTemporaryFile(
            mode='w+', prefix='ffmpeg_', suffix='.log', delete=False
        )
        print(f"  FFmpeg log: {self._ffmpeg_stderr.name}")

        self._ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._ffmpeg_stderr
        )

        if self._ffmpeg_process.poll() is not None:
            raise RuntimeError("FFmpeg process failed to start")

    def _write(self, image: np.ndarray, name: str):
        '''
        This is a convenience wrapper for writing images to disk.
        '''
        if _READ_WRITE_FROM_TO_BUFFER:
            self.buff.append(image)
            return

        if self.path.lower().endswith(self.extension.lower()):
            path = self.path
        else:
            name = f'{name}{self.suffix}.{self.extension}'
            path = os.path.join(self.path, name)

        # Use FFmpeg for all video files (consistent approach)
        is_video = self.extension.lower() in ['mp4', 'mov']

        if is_video:
            # All video output uses FFmpeg subprocess
            if self._ffmpeg_process is None:
                h, w = image.shape[:2]
                self._setup_ffmpeg_writer(w, h, path)

            # Scale image to appropriate bit depth
            if self.bit_depth > 8:
                frame_data = (image * 65535).astype(np.uint16)
            else:
                frame_data = (image * 255).astype(np.uint8)

            # Write frame
            try:
                self._ffmpeg_process.stdin.write(frame_data.tobytes())
                self._ffmpeg_process.stdin.flush()
                self._frame_count += 1
                if self._frame_count % 100 == 0:
                    print(f"  Wrote {self._frame_count} frames...")
            except BrokenPipeError:
                raise RuntimeError("FFmpeg process died unexpectedly")

        else:
            # Non-video files (images)
            # save handles rescaling for us
            if self.separate_bands:
                paths = _get_band_paths(path, image.shape[2])
                for b, path in enumerate(paths):
                    if os.path.exists(path):
                        raise FileExistsError(path)
                    save(image[..., b], path)
            else:
                if os.path.exists(path):
                    raise FileExistsError(path)
                save(image, path)

    def release(self):
        '''
        Releases the writer after you're finished with it. This has no effect
        if you're writing individual frames at a time, but is necessary when
        writing video.
        '''
        # Release FFmpeg process if active
        if self._ffmpeg_process is not None:
            try:
                self._ffmpeg_process.stdin.close()
                self._ffmpeg_process.wait(timeout=30)
                if self._frame_count > 0:
                    audio_msg = " (with audio)" if (
                        self.include_audio and
                        self.audio_source is not None and
                        os.path.exists(self.audio_source)
                    ) else ""
                    msg = f"  Completed: {self._frame_count}"
                    print(f"{msg} frames{audio_msg}")
            except subprocess.TimeoutExpired:
                self._ffmpeg_process.kill()
                print("  Warning: FFmpeg process timed out")
            finally:
                if self._ffmpeg_stderr is not None:
                    self._ffmpeg_stderr.close()
                self._ffmpeg_process = None

        # Release OpenCV writers
        if self._writer is not None:
            for writer in self._writer:
                writer.release()

    def reset(self):
        '''
        Resets the :class:`Writer` to the start.
        '''
        # TODO: Should this delete or otherwise clear out anything that's been
        # written out previously? The expectation would be that Writer should
        # only receive HoldTokens prior to reset being called, but that cannot
        # be guaranteed.
        self._writer = None
        self._ffmpeg_process = None
        self._frame_count = 0

    def set_path(self, path: Optional[str]):
        '''
        Sets the output path.
        '''
        if path is not None:
            if path.lower().endswith(self.extension.lower()):
                root = os.path.dirname(path)
            else:
                root = path
            os.makedirs(root, exist_ok=True)

        self.path = path


class OutOfInputs(Exception):
    '''
    This exception is raised by a :class:`Loader` when it exhausts its
    available inputs.
    '''


class MisshapenImageError(Exception):
    '''
    This exception is raised by a :class:`Loader` when it attempts to load an
    image or video, but finds it has a different shape from what is expected.
    '''


def _get_band_paths(path: str, num_bands: int) -> List[str]:
    '''
    Convenience function for creating paths for separate bands from a single
    path.
    '''
    base_path, ext = os.path.splitext(path)
    return [f'{base_path}_{b}{ext}' for b in range(num_bands)]


def _get_num_frames(r: Optional[cv2.VideoCapture]) -> int:
    '''
    Convenience function for calculating number of frames in an MP4.
    '''
    if r is None:
        return 1
    else:
        return r.get(cv2.CAP_PROP_FRAME_COUNT)


def trim_video(input_path: str, output_path: str,
               start_frame: int = 0,
               max_frames: Optional[int] = None) -> int:
    '''
    Trim a video file by extracting a range of frames.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to write the trimmed video.
        start_frame (int): Frame index to start from (default: 0).
        max_frames (optional, int): Maximum number of frames to write.
        If None, writes all frames from start_frame to end.

    Returns:
        int: Number of frames written.
    '''
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return frame_count


_LUT = np.arange(0, 256, dtype=np.float32) / 256.


def _convert_and_scale_uint8(image: np.ndarray,
                             out: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Convenience function for converting a :class:`numpy.ndarray` of dtype uint8
    into float32 and scaling by 256.
    '''
    # cv2.LUT is parallelized. However, the parallelization is shape-dependent.
    # From experimentation, the shape (1, -1) is consistently quite speedy, and
    # it works with arbitrarily-shaped input images.
    if image.dtype == np.uint8:
        out = cv2.LUT(
            image.reshape(1, -1),
            _LUT,
            None if (out is None) else out.reshape(1, -1),
        )
        return out.reshape(*image.shape)
    else:
        return np.divide(image.astype(np.float32), 256., out=out)


@contextmanager
def _read_write_from_to_buffer():
    '''
    Context manager inside of which :class:`video2vision.Loader` and
    :class:`video2vision.Writer` will read from and write to internal buffers
    instead of disk.
    '''
    global _READ_WRITE_FROM_TO_BUFFER
    _READ_WRITE_FROM_TO_BUFFER, was = True, _READ_WRITE_FROM_TO_BUFFER
    try:
        yield
    finally:
        _READ_WRITE_FROM_TO_BUFFER = was
