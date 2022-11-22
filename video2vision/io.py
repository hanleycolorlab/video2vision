from contextlib import contextmanager
from glob import glob
import os
from typing import Dict, Iterator, List, Optional, Union

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

__all__ = ['load', 'Loader', 'OutOfInputs', 'save', 'Writer']

# Default video codec
_VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

_VIDEO_FPS = 23.976023976023978

# This is a global variable used to track whether the Loader/Writer should
# read/write from disk or to an internal buffer. This internal buffer is used
# to apply a Pipeline to images in memory instead of running the normal way.
_READ_WRITE_FROM_TO_BUFFER = False

_IMAGE_EXTENSIONS = ['arw', 'jpeg', 'jpg', 'mp4', 'png', 'raw', 'tif', 'tiff']


def load(path: str) -> np.ndarray:
    '''
    Convenience function for loading images from disk. Dispatches to
    appropriate backend.

    Args:
        path (str): Path to image to load.
    '''
    # Some of the function calls below don't raise an error if the file doesn't
    # exist, they just fail silently. So let's check explicitly.
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(('.tif', '.tiff')):
        if not has_tiff:
            raise ImportError('tifffile is needed to read tif files')
        image = tifffile.imread(path)

    elif path.lower().endswith('.mp4'):
        reader = cv2.VideoCapture(path)
        frames, ret = [], True
        while ret and reader.isOpened():
            ret, frame = reader.read()
            if ret:
                frames.append(frame)
        reader.release()
        image = np.stack(frames, axis=2)

    elif path.lower().endswith('.arw'):
        if not has_rawpy:
            raise ImportError('rawpy is needed to read arw files')
        with rawpy.imread(path) as raw_file:
            image = raw_file.postprocess()

    else:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Rescale to [0, 1] and float32
    image = image.astype(np.float32) / 256.

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

    if path.lower().endswith('.mp4') and image.ndim == 4:
        # MP4 only likes uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        # Arguments are path, codec, frame rate, size, is_color
        writer = cv2.VideoWriter(
            path,
            _VIDEO_CODEC,
            _VIDEO_FPS,
            image.shape[:2][::-1],
            (image.shape[-1] > 1),
        )
        for t in range(image.shape[2]):
            writer.write(image[:, :, t, :])
        writer.release()

    elif image.ndim == 4:
        raise NotImplementedError('Video is supported only for mp4')

    elif path.lower().endswith(('.tif', '.tiff')):
        if not has_tiff:
            raise ImportError('tifffile is needed to write tif files')
        # photometric='minisblack' suppresses a DeprecationWarning.
        tifffile.imwrite(path, image, photometric='minisblack')

    else:
        return cv2.imwrite(path, image)


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

    def __init__(self, path: Optional[Union[str, Iterator[str]]] = None,
                 batch_size: int = 1):
        '''
        Args:
            path (str or iterator of str): Path(s) to the images to load. This
            can be a directory, in which case everything in the directory will
            be loaded, or it can be a path to a single image, or it can be a
            string with wildcards.

            batch_size (int): Number of images to return at a time.
        '''
        self.set_path(path)
        self.batch_size = batch_size
        # This is used to provide external inputs from memory to a pipeline,
        # instead of loading from disk. It is only consulted if the global
        # variable _READ_WRITE_FROM_BUFFER is true.
        self.buff: List[np.ndarray] = []

    def __iter__(self) -> Iterator[np.ndarray]:
        # If _READ_WRITE_FROM_BUFFER, we are providing inputs from memory, not
        # loading from disk. This is used to apply Pipelines to images in
        # memory.
        if _READ_WRITE_FROM_TO_BUFFER:
            while self.buff:
                yield self.buff.pop(0), None

        else:
            for path, reader in zip(self.paths, self._readers):
                name, _ = os.path.splitext(os.path.basename(path))
                if reader is not None:
                    ret, frame = reader.read()
                    while ret and reader.isOpened():
                        # Rescale to [0, 1] before returning
                        yield frame.astype(np.float32) / 256., name
                        ret, frame = reader.read()
                else:
                    # load handles rescaling for us
                    yield load(path), name

    def __len__(self) -> int:
        if _READ_WRITE_FROM_TO_BUFFER:
            return len(self.buff)
        else:
            return sum(int(_get_num_frames(r)) for r in self._readers)

    def apply(self) -> Dict:
        # If _READ_WRITE_FROM_BUFFER, we are providing inputs from memory, not
        # loading from disk. This is used to apply Pipelines to images in
        # memory.
        if _READ_WRITE_FROM_TO_BUFFER:
            if self.buff:
                out, self.buff = np.stack(self.buff, axis=2), []
                return {'image': out}
            else:
                raise OutOfInputs('Buffer is empty')

        try:
            frame, name = next(self._data_iter)
        except StopIteration:
            raise OutOfInputs()
        names = [name]

        if self.batch_size == 1:
            return {'image': frame, 'names': names}

        else:
            h, w, c = frame.shape
            rtn = np.empty((h, w, self.batch_size, c), dtype=frame.dtype)
            rtn[:, :, 0, :] = frame

            try:
                for t in range(1, self.batch_size):
                    rtn[:, :, t, :], name = next(self._data_iter)
                    names.append(name)
            except StopIteration:
                rtn = rtn[:, :, :t, :]

            return {'image': rtn, 'names': names}

    def get_frame(self, t: int) -> np.ndarray:
        '''
        Retrieves a single frame and returns it.
        '''
        if _READ_WRITE_FROM_TO_BUFFER:
            return self.buff[t]

        n_frames = 0
        for path, reader in zip(self.paths, self._readers):
            n_frames += _get_num_frames(reader)
            if n_frames > t:
                break
        else:
            raise ValueError(f'{t} out of range: {len(self)}')

        if reader is None:
            # load handles rescaling for us
            image = load(path)
        else:
            reader.set(cv2.CAP_PROP_POS_FRAMES, t)
            _, image = reader.read()
            image = image.astype(np.float32) / 256.

        return image

    def set_path(self, paths: Optional[Union[str, Iterator[str]]]):
        '''
        Sets the input path.
        '''
        if paths is None:
            self.paths, self._readers = None, []
        else:
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

        self._data_iter = iter(self)

    def _to_json(self) -> Dict:
        return {
            'class': self.__class__.__name__,
            'batch_size': self.batch_size,
        }


@OPERATOR_REGISTRY.register
class Writer(Operator):
    '''
    This is an :class:`Operator` that writes finished images to disk. It acts
    as a leaf of the :class:`Pipeline` DAG.
    '''

    def __init__(self, path: Optional[str] = None,
                 extension: Optional[str] = None,
                 separate_bands: bool = False, suffix: str = ''):
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
        '''
        if extension is None:
            # Check for extension in path
            _, extension = os.path.splitext(path or '')
            extension = extension[1:].lower()
            if extension not in _IMAGE_EXTENSIONS:
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

    def apply(self, x: Dict):
        image = x['image']
        num_frames = 1 if (image.ndim == 3) else image.shape[2]
        names = x.get('names', [None] * num_frames)

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

        # TODO: This should be handled better.
        if self.extension.lower() == 'mp4':
            # Rescale to [0, 256], and MP4 only likes uint8
            image = np.clip(256 * image, 0, 255).astype(np.uint8)

            if self.separate_bands:
                if self._writer is None:
                    paths = _get_band_paths(path, image.shape[2])
                    if any(os.path.exists(p) for p in paths):
                        raise FileExistsError(paths)
                    self._writer = [
                        _get_writer(p, image[..., 0:1]) for p in paths
                    ]
                if len(self._writer) != image.shape[2]:
                    raise RuntimeError(
                        f'Mismatch in number of bands: {len(self._writer)} '
                        f'vs {image.shape[2]}'
                    )
                for b, writer in enumerate(self._writer):
                    writer.write(image[..., b])
            else:
                if self._writer is None:
                    self._writer = [_get_writer(path, image)]
                self._writer[0].write(image)

        else:
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
        if self._writer is not None:
            for writer in self._writer:
                writer.release()

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


def _get_writer(path: str, image: np.ndarray) -> cv2.VideoWriter:
    '''
    Convenience function for building an MP4 writer.
    '''
    return cv2.VideoWriter(
        path,
        _VIDEO_CODEC,
        _VIDEO_FPS,
        image.shape[:2][::-1],
        (image.shape[-1] > 1),
    )


@contextmanager
def _read_write_from_to_buffer():
    '''
    Context manager inside of which :class:`video2vision.Loader` and
    :class:`video2vision.Writer` will read from and write to internal buffers instead of disk.
    '''
    global _READ_WRITE_FROM_TO_BUFFER
    _READ_WRITE_FROM_TO_BUFFER, was = True, _READ_WRITE_FROM_TO_BUFFER
    try:
        yield
    finally:
        _READ_WRITE_FROM_TO_BUFFER = was
