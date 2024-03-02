from contextlib import contextmanager
from functools import lru_cache
import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import ipyevents as events
import ipywidgets as widgets
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traitlets

import video2vision as v2v

from .utils import gamma_scale, resize

__all__ = [
    'DisplayBox', 'ForwardBackwardButtons', 'GhostBox', 'SelectorBox',
]

ASSOCIATION_RADIUS_SQ = 32**2

_PAIR = traitlets.Tuple(traitlets.Int(), traitlets.Int())

DEFAULT_CROSSHAIR = np.zeros((10, 10, 4), dtype=np.uint8)
DEFAULT_CROSSHAIR[:, :, 1] = DEFAULT_CROSSHAIR[:, :, 3] = 255

# Try to get Inconsolata. If it's not available, try arial, which should be
# available on most Windows systems.
try:
    ImageFont.truetype('Inconsolata.otf', 32)
except OSError:
    FONT = 'Arial.ttf'
else:
    FONT = 'Inconsolata.otf'


FONT_COLOR = (0, 255, 0)

# TODO: Combine DisplayBox, SelectorBox


class DisplayBox(widgets.VBox):
    '''
    This is a widget for displaying images from two or more different
    :class:`v2v.Loader`s side by side. It provides buttons for paging forward
    or backward in a video or set of images, but no other functionality.
    '''
    def __init__(self, *loaders, t: int = 0,
                 shifts: Optional[List[int]] = None,
                 output_size: Optional[Union[float, Tuple[int, int]]] = None):
        '''
        Args:
            loaders (:class:`video2vision.Loader`): The loaders to use.
            t (int): Initial time value.
            shifts (optional, list of int): If provided, this gives offsets to
            apply to the loaders relative to each other, e.g. if there is a
            temporal shift between the sources.
            output_size (optional, float or pair of int): If provided, the box
            will have this size display. A floating point value is interpreted
            as a proportional resize.
        '''
        self.loaders = loaders
        if shifts is None:
            shifts = [0 for _ in self.loaders]
        self.shifts = shifts

        # Set fixed output size
        if output_size is None:
            self.h = max(r.expected_size[1] for r in self.loaders)
            self.w = sum(r.expected_size[0] for r in self.loaders)
        elif isinstance(output_size, float):
            h = max(r.expected_size[1] for r in self.loaders)
            w = sum(r.expected_size[0] for r in self.loaders)
            self.h, self.w = int(output_size * h), int(output_size * w)
        else:
            self.w, self.h = output_size

        # Determine allowed values for t
        num_frames = min(len(loader) for loader in self.loaders)
        min_t, max_t = -min(shifts), num_frames - max(shifts)

        # Construct widgets
        self.display = Image.new('RGB', (self.w, self.h))
        self.buttons = ForwardBackwardButtons(self.set_frame, min_t, max_t, t)
        self.display_image = widgets.Image(format='png')
        super().__init__((self.display_image, self.buttons))

        # Display first image
        self.set_frame(t)

    def set_frame(self, t: int):
        '''
        Sets the current frame.
        '''
        with self.buttons.disable():
            images = [
                r.get_frame(t + s) for r, s in zip(self.loaders, self.shifts)
            ]
            self.set_images(*images)

    def set_images(self, *images: np.ndarray):
        '''
        Sets the current image(s).
        '''
        # Apply gamma scaling to ensure visually correct display
        images = [gamma_scale(image) for image in images]
        im_w, im_h = self.w // len(images), self.h

        x = 0
        for image in images:
            # Convert from BGR, float32, [0, 1] to RGB, uint8, [0, 255]
            if (image.ndim == 3) and (image.shape[2] == 3):
                image = image[:, :, ::-1]
            image = np.clip(256 * image, 0, 255).astype(np.uint8)
            # Resize
            if image.size != (im_w, im_h):
                image = resize(image, (im_w, im_h))
            self.display.paste(Image.fromarray(image), (x, 0))
            x += image.shape[1]
        self.display_image.value = self.display._repr_png_()

    @property
    def t(self) -> int:
        return self.buttons.t

    @t.setter
    def t(self, t: int):
        self.buttons.t = t


class ForwardBackwardButtons(widgets.HBox):
    '''
    This widget provides a set of buttons for paging forward and backward in a
    video or set of images.
    '''
    def __init__(self, call_func: Callable, min_t: int, max_t: int,
                 t: int = 0):
        self.t, self.min_t, self.max_t = t, min_t, max_t
        self._call_func = call_func

        buttons = []
        for shift in [-100, -10, -1, 1, 10, 100]:
            buttons.append(widgets.Button(description=f'{shift:+}'))
            buttons[-1].on_click(self._get_call_func(shift))

        super().__init__(buttons)

    @contextmanager
    def disable(self):
        for button in self.children:
            button.disabled = True
        yield
        for button in self.children:
            button.disabled = False

    def _get_call_func(self, shift: int):
        def on_click(b):
            self.t = max(min(self.t + shift, self.max_t), self.min_t)
            self._call_func(self.t)
        return on_click


class GhostBox(DisplayBox):
    '''
    This widget overlays two images on each other, using one to provide the
    red and blue channels and the other to provide the green channel. This is
    used to test the alignment of two images.
    '''
    def __init__(self, loader_0: v2v.Loader, loader_1: v2v.Loader, t: int = 0,
                 shifts: Optional[List[int]] = None,
                 output_size: Optional[Union[float, Tuple[int, int]]] = None):
        '''
        Args:
            loader_0 (:class:`video2vision.Loader`): The loader to use for the
            red and blue bands.
            loader_1 (:class:`video2vision.Loader`): The loader to use for the
            green band.
            t (int): Initial time value.
            shifts (optional, list of int): If provided, this gives offsets to
            apply to the loaders relative to each other, e.g. if there is a
            temporal shift between the sources.
            output_size (optional, float or pair of int): If provided, the box
            will have this size display. A floating point value is interpreted
            as a proportional resize.
        '''
        # This is needed to prevent DisplayBox.__init__ from assuming that we
        # will display both side-by-side instead of overlaid when calculating
        # height and width.
        if output_size is None:
            output_size = 1.
        if isinstance(output_size, float):
            h = max(loader_0.expected_size[1], loader_1.expected_size[1])
            w = max(loader_0.expected_size[0], loader_1.expected_size[0])
            output_size = (int(output_size * w), int(output_size * h))
        super().__init__(
            loader_0, loader_1, t=t, shifts=shifts, output_size=output_size
        )

    def set_images(self, rgb_image: np.ndarray, uv_image: np.ndarray):
        rgb_image[:, :, 1] = uv_image[:, :, 2]
        super().set_images(rgb_image)


class SelectorBox(DisplayBox):
    '''
    This widget allows the user to select points on an image. It is intended to
    be used to select sample points.
    '''
    def __init__(self, loader: v2v.Loader, t: int = 0, w: int = 25,
                 align_pipeline: Optional[v2v.Pipeline] = None,
                 box_color: np.ndarray = (0, 255, 0),
                 font_color: Optional[np.ndarray] = None,
                 auto_op: Optional[Union[str, v2v.AutoLinearize]] = None,
                 cache_path: Optional[str] = None,
                 copy_from: Optional['SelectorBox'] = None,
                 border_margin: int = 10,
                 output_size: Optional[Union[float, Tuple[int, int]]] = None):
        '''
        Args:
            loader (:class:`video2vision.Loader`): Loader pointing to source of
            images. Unlike :class:`DisplayBox`, this takes only a single
            loader.
            t (int): Initial time.
            w (int): Width of selector box in pixels.
            align_pipeline (optional, :class:`v2v.Pipeline`): If provided, use
            this to align the image prior to display.
            box_color (:class:`numpy.ndarray`): Color of the selector boxes.
            font_color (optional, :class:`numpy.ndarray`): Color of the numbers
            next to the sample boxes. If not provided, equal to box_color.
            auto_op (optional, str or :class:`video2vision.AutoLinearize`): If
            provided, use this autolinearizer to automatically locate sample
            locations. If this is a string, assume it is the path to a saved
            :class:`video2vision.AutoLinearize`.
            cache_path (optional, str): If provided, use this location to cache
            the location of the crosshairs.
            copy_from (optional, :class:`SelectorBox`): If provided, copy the
            initial sample locations from this.
            output_size (optional, pair of int): If provided, the box will have
            this size display.
        '''
        self.align_pipeline = align_pipeline
        self.font_color = font_color or box_color
        self.box_color = np.array(box_color)
        self.border_margin = border_margin
        self.sample_size = w

        # These will store the current crosshairs:
        # idxs (List[int]): Numerical indices of the crosshairs.
        # crosshair_type (List[int]): Whether they are Xs or boxes.
        # crosshairs (List[Tuple[int, int]]): The actual locations of the
        # centers of the crosshairs, in the original image coordinates, not the
        # display coordinates.
        self.idxs, self.crosshair_type, self.crosshairs = [], [], []

        if (cache_path is not None) and os.path.isdir(cache_path):
            cache_path = os.path.join(cache_path, 'crosshairs.json')
        self.cache_path = cache_path

        if (auto_op is not None) and isinstance(auto_op, str):
            auto_op = v2v.load_operator(auto_op)

        # The order here is important. In terms of precedence, from most to
        # least, we have:
        #
        # 1. The cached crosshairs.
        # 2. The crosshairs in the SampleSelector provided as the copy_from
        #    argument.
        # 3. The crosshairs found by the auto-localizer.
        #
        # When we call super.__init__, that will call set_frame, which will
        # call _autofind_crosshairs if self.auto_op is available. Therefore, if
        # either 1 or 2 is available, we leave self.auto_op set to None until
        # after we call super.__init__. Otherwise, we set it before.

        self.auto_op = None
        if (self.cache_path is not None) and os.path.exists(self.cache_path):
            t = self.load_crosshairs(self.cache_path, norefresh=True)
        elif copy_from is not None:
            self.idxs = copy_from.idxs
            self.crosshair_type = copy_from.crosshair_type
            self.crosshairs = copy_from.crosshairs
            t += copy_from.t - copy_from.shifts[0]
        else:
            self.auto_op = auto_op

        super().__init__(loader, t=t, output_size=output_size)

        self.auto_op = auto_op

        # Register method for handling clicks
        self.dom_handler = events.Event(
            source=self.display_image, watched_events=['click']
        )
        self.dom_handler.on_dom_event(self._handle_click)

    def _autofind_crosshairs(self, image):
        if self.auto_op is not None:
            try:
                crosshairs, _ = self.auto_op._locate_samples(image)
            except Exception:
                self.idxs, self.crosshair_type, self.crosshairs = [], [], []
            else:
                self.idxs = list(range(crosshairs.shape[0]))
                self.crosshair_type = [0 for _ in self.idxs]
                self.crosshairs = crosshairs.astype(np.int64).tolist()

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        if max(self.idxs) + 1 != len(self.idxs):
            raise RuntimeError('Not all samples selected')

        samples = v2v.utils.extract_samples(
            self._original_image, self.crosshairs, self.sample_size
        )
        types = np.array(self.crosshair_type, dtype=bool)
        reidx = [self.idxs.index(i) for i in range(samples.shape[0])]

        return samples[reidx, :], types[reidx]

    def _handle_click(self, event: Dict):
        # Extract point from event dictionary
        x, y = event['dataX'], event['dataY']
        # Get rescaling factor
        h, w, *_ = self._cached_image.shape
        rs = (self.original_size[0] / w, self.original_size[1] / h)

        # If self.crosshairs is empty, the argmin will error out.
        if self.crosshairs:
            # We want to do the comparison in the displayed coordinate space,
            # not the original coordinate system.
            disp_ch = np.array(self.crosshairs) / np.array(rs)
            dist_sq = ((disp_ch - np.array([x, y]))**2).sum(1)
            min_idx = np.argmin(dist_sq)
            min_dist_sq = dist_sq[min_idx]
        else:
            min_dist_sq = ASSOCIATION_RADIUS_SQ + 1

        if min_dist_sq <= ASSOCIATION_RADIUS_SQ:
            self.idxs.pop(min_idx)
            self.crosshair_type.pop(min_idx)
            self.crosshairs.pop(min_idx)
        else:
            # Rescale point to original coordinate system
            x, y = int(rs[0] * x), int(rs[1] * y)
            # Check if we're too close to the border
            if (
                (min(x, self.original_size[0] - x) < self.border_margin) or
                (min(y, self.original_size[1] - y) < self.border_margin)
            ):
                return

            # Get the next index to be assigned by looking for the lowest index
            # that has not yet been used.
            if self.idxs:
                idx = min({i + 1 for i in self.idxs} - set(self.idxs))
            else:
                idx = 0
            self.idxs.append(idx)
            self.crosshair_type.append(int(event['shiftKey']))
            self.crosshairs.append((x, y))

        if self.cache_path is not None:
            self.save_crosshairs(self.cache_path)

        self._update_image()

    def load_crosshairs(self, path: str, norefresh: bool = False):
        with open(path, 'r') as crosshairs_file:
            crosshairs = json.load(crosshairs_file)

        self.idxs = crosshairs['idxs']
        self.crosshair_type = crosshairs['crosshair_type']
        self.crosshairs = crosshairs['crosshairs']

        if not norefresh:
            if crosshairs['t'] != self.t:
                self.set_frame(crosshairs['t'], noauto=True)
            else:
                self._update_image()
        else:
            return crosshairs['t']

    @lru_cache
    def make_crosshairs(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        r_w, r_h = max(1, w // 25), max(1, h // 25)
        box_color = np.concatenate((self.box_color, (1,)))

        crosshair_image = np.zeros((h, w, 4), dtype=np.uint8)
        crosshair_image[:r_h, :, :] = crosshair_image[-r_h:, :, :] = box_color
        crosshair_image[:, :r_w, :] = crosshair_image[:, -r_w:, :] = box_color

        shift_image = np.zeros((h, w, 4), dtype=np.uint8)
        shift_image = Image.fromarray(shift_image)
        draw = ImageDraw.Draw(shift_image)
        draw.line((0, 0, w, h), fill=tuple(box_color))
        draw.line((0, h, w, 0), fill=tuple(box_color))
        shift_image = np.array(shift_image)

        return crosshair_image, shift_image

    def save_crosshairs(self, path: str):
        crosshairs = {
            'idxs': self.idxs,
            'crosshair_type': self.crosshair_type,
            'crosshairs': self.crosshairs,
            't': self.t,
        }

        with open(path, 'w') as crosshairs_file:
            json.dump(crosshairs, crosshairs_file)

    def set_frame(self, t: int, noauto: bool = False):
        '''
        Sets the current frame.
        '''
        with self.buttons.disable():
            self.t = t
            image = self.loaders[0].get_frame(t) + self.shifts[0]
            if not noauto:
                self._autofind_crosshairs(image)
            self.set_images(image)

    def set_images(self, image: np.ndarray):
        if self.align_pipeline is not None:
            image = self.align_pipeline(image, np.empty_like(image))
        # Cache original image to use in retrieving samples
        self._original_image = image

        # Apply gamma scaling to ensure visually correct display
        image = gamma_scale(image)

        # Convert from BGR, float32, [0, 1] to RGB, uint8, [0, 255]
        if (image.ndim == 3) and (image.shape[2] == 3):
            image = image[:, :, ::-1]
        image = np.clip(256 * image, 0, 255).astype(np.uint8)

        ch_h = int(self.sample_size * self.h / image.shape[0])
        ch_w = int(self.sample_size * self.w / image.shape[1])
        self._cached_crosshairs = self.make_crosshairs(ch_w, ch_h)

        self.original_size = image.shape[:2][::-1]
        if (self.h, self.w) != image.shape[:2]:
            image = resize(image, (self.w, self.h))

        self._cached_image = image
        self._update_image()

    def _update_cache(self):
        if self.cache_path is not None:
            self.save_crosshairs(self.cache_path)

    def _update_image(self):
        # We're going to modify the image in place, so clone it. It should
        # already be resized to the correct size.
        image = np.copy(self._cached_image)

        h, w, *_ = image.shape
        rs = (w / self.original_size[0], h / self.original_size[1])

        ch_h, ch_w, _ = self._cached_crosshairs[0].shape

        for (x, y), s in zip(self.crosshairs, self.crosshair_type):
            patch = self._cached_crosshairs[s]
            mask, patch = patch[:, :, 3:4], patch[:, :, :3]
            ul_x = max(int(x * rs[0]) - (ch_w // 2), 0)
            ul_y = max(int(y * rs[1]) - (ch_h // 2), 0)
            lr_x, lr_y = ul_x + patch.shape[1], ul_y + patch.shape[0]
            image[ul_y:lr_y, ul_x:lr_x, :] = (
                (image[ul_y:lr_y, ul_x:lr_x, :] * (1 - mask)) +
                (patch[:, :, :3] * mask)
            )

        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        for number, (x, y) in zip(self.idxs, self.crosshairs):
            x = int(x * rs[0]) + (ch_w // 2) + 4
            y = int(y * rs[1]) + (ch_h // 2) + 4
            draw.text(
                (x, y), str(number),
                font=ImageFont.truetype(FONT, int(32 * w / 2000)),
                fill=self.font_color
            )

        self.display = image
        self.display_image.value = self.display._repr_png_()
