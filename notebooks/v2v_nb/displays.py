from contextlib import contextmanager
from functools import lru_cache
import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
from ipycanvas import MultiCanvas
import ipyevents as events
import ipywidgets as widgets
import numpy as np
from PIL import Image, ImageDraw
import traitlets

import video2vision as v2v

from .choices import SimpleButton

__all__ = ['DisplayBox', 'GhostBox', 'SelectorBox']

ASSOCIATION_RADIUS_SQ = 4**2

_PAIR = traitlets.Tuple(traitlets.Int(), traitlets.Int())

DEFAULT_CROSSHAIR = np.zeros((10, 10, 4), dtype=np.uint8)
DEFAULT_CROSSHAIR[:, :, 1] = DEFAULT_CROSSHAIR[:, :, 3] = 255


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

        # Construct widgets
        self.buttons = self.make_button_panel(t)
        self.canvas = MultiCanvas(2, width=self.w, height=self.h, sync_image_data=True)
        super().__init__((self.canvas, self.buttons))

        # Display first image
        self.set_frame(t)

    def make_button_panel(self, t: int = 0) -> widgets.Widget:
        # Break this out as a method so it can be overridden by subclasses.
        num_frames = min(len(loader) for loader in self.loaders)
        min_t, max_t = -min(self.shifts), num_frames - max(self.shifts)
        return ButtonPanel(self.set_frame, min_t, max_t, t)

    def set_frame(self, t: int):
        '''
        Sets the current frame.
        '''
        with self.buttons.disable():
            images = [
                r.get_frame(t + s, for_display=True)
                for r, s in zip(self.loaders, self.shifts)
            ]
            self.set_images(*images)

    def set_images(self, *images: np.ndarray):
        '''
        Sets the current image(s).
        '''
        # Apply gamma scaling to ensure visually correct display
        im_w, im_h = self.w // len(images), self.h
        self.message = 0

        x = 0
        for image in images:
            # Convert from BGR, float32 to RGB, uint8
            if (image.ndim == 3) and (image.shape[2] == 3):
                image = image[:, :, ::-1]
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.shape[:2] != (im_h, im_w):
                image = cv2.resize(image, (im_w, im_h))
            self.canvas[0].put_image_data(image, x, 0)
            x += image.shape[1]

    @property
    def t(self) -> int:
        return self.buttons.t

    @t.setter
    def t(self, t: int):
        self.buttons.t = t


class ButtonPanel(widgets.HBox):
    '''
    This widget provides a set of buttons for paging forward and backward in a
    video or set of images.
    '''
    def __init__(self, call_func: Callable, min_t: int, max_t: int,
                 t: int = 0, clear_func: Optional[Callable] = None):
        self.t, self.min_t, self.max_t = t, min_t, max_t
        self._call_func = call_func

        buttons = []
        for shift in [-100, -10, -1, 1, 10, 100]:
            buttons.append(widgets.Button(description=f'{shift:+}'))
            buttons[-1].on_click(self._get_call_func(shift))

        if clear_func is not None:
            buttons.append(SimpleButton('Clear All', clear_func))

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
            loader_0, loader_1, t=t, shifts=shifts, output_size=output_size,
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
                 marker_choice: str = 'box',
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
            marker_choice (str): Choice of type of markers to place when
            clicked. Options: 'box', 'cross'.
            output_size (optional, pair of int): If provided, the box will have
            this size display.
        '''
        # TODO: Something broke when w was too big...
        if marker_choice not in {'box', 'cross'}:
            raise ValueError(marker_choice)

        self.align_pipeline = align_pipeline
        self.font_color = font_color or box_color
        self.border_margin = border_margin
        self.sample_size = w
        self.marker_choice = marker_choice

        # These will store the current crosshairs:
        # idxs (List[int]): Numerical indices of the crosshairs.
        # crosshair_type (List[int]): Whether they are Xs or boxes.
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

        box_color = f"#{box_color[0]:02x}{box_color[1]:02x}{box_color[2]:02x}"
        self.canvas[1].stroke_style = box_color

        self.auto_op = auto_op

        # Register method for handling clicks
        self.dom_handler = events.Event(
            source=self.canvas, watched_events=['click']
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

    def clear_crosshairs(self):
        self.idxs = []
        self.crosshair_type = []
        self.crosshairs = []

        if self.cache_path is not None:
            self.save_crosshairs(self.cache_path)

        self.canvas[1].clear_rect(0, 0, self.w, self.h)

    def _draw_crosshair(self, pt: Tuple[int, int], crosshair_type: bool):
        # Rescale to display coordinate system
        x = int(self.w * pt[0] / self.original_size[0])
        y = int(self.h * pt[1] / self.original_size[1])
        ch_w = max(int(self.sample_size * self.w / self.original_size[0]), 1)
        ch_h = max(int(self.sample_size * self.h / self.original_size[1]), 1)
        ul_x, ul_y = x - (ch_w // 2), y - (ch_h // 2)
        lr_x, lr_y = ul_x + ch_w, ul_y + ch_h

        if self.marker_choice == 'cross':
            if crosshair_type:
                self.canvas[1].stroke_line(ul_x, ul_y, lr_x, lr_y)
                self.canvas[1].stroke_line(ul_x, lr_y, lr_x, ul_y)
            else:
                self.canvas[1].stroke_line(ul_x, y, lr_x, y)
                self.canvas[1].stroke_line(x, ul_y, x, lr_y)
        elif self.marker_choice == 'box':
            self.canvas[1].stroke_rect(ul_x, ul_y, ch_w, ch_h)

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.idxs) == 0:
            return np.empty((0, 3)), np.empty((0,), dtype=bool)

        if max(self.idxs) + 1 != len(self.idxs):
            raise RuntimeError('Not all samples selected')

        image = self.loaders[0].get_frame(
            self.t + self.shifts[0], for_display=False,
        )
        if self.align_pipeline is not None:
            image = self.align_pipeline(image, np.empty_like(image))
        samples = v2v.utils.extract_samples(
            image, self.crosshairs, self.sample_size
        )
        types = np.array(self.crosshair_type, dtype=bool)
        reidx = [self.idxs.index(i) for i in range(samples.shape[0])]

        return samples[reidx, :], types[reidx]

    def _handle_click(self, event: Dict):
        self.event = event
        # Extract point from event dictionary
        w, h = self.original_size
        x = int(w * event['relativeX'] / event['boundingRectWidth'])
        y = int(h * event['relativeY'] / event['boundingRectHeight'])

        # If self.crosshairs is empty, the argmin will error out.
        if self.crosshairs:
            # We want to do the comparison in the displayed coordinate space,
            # not the original coordinate system.
            disp_ch = np.array(self.crosshairs)
            dist_sq = ((disp_ch - np.array([x, y]))**2).sum(1)
            min_idx = np.argmin(dist_sq)
            min_dist_sq = dist_sq[min_idx]
        else:
            min_dist_sq = ASSOCIATION_RADIUS_SQ + 1

        if min_dist_sq <= ASSOCIATION_RADIUS_SQ:
            self._remove_crosshair(self.crosshairs[min_idx])
            self.idxs.pop(min_idx)
            self.crosshair_type.pop(min_idx)
            self.crosshairs.pop(min_idx)
        else:
            # Check if we're too close to the border
            if (
                (min(x, self.original_size[0] - x) < self.border_margin) or
                (min(y, self.original_size[1] - y) < self.border_margin)
            ):
                return

            # Get the next index to be assigned by looking for the lowest index
            # that has not yet been used.
            if self.idxs:
                idx = min(({0} | {i + 1 for i in self.idxs}) - set(self.idxs))
            else:
                idx = 0
            self.idxs.append(idx)
            self._draw_crosshair((x, y), int(event['shiftKey']))
            self.crosshair_type.append(int(event['shiftKey']))
            self.crosshairs.append((x, y))

        if self.cache_path is not None:
            self.save_crosshairs(self.cache_path)

    def load_crosshairs(self, path: str, norefresh: bool = False):
        with open(path, 'r') as crosshairs_file:
            crosshairs = json.load(crosshairs_file)

        self.idxs = crosshairs['idxs']
        self.crosshair_type = crosshairs['crosshair_type']
        # JSON turns the pairs into lists
        self.crosshairs = [tuple(x) for x in crosshairs['crosshairs']]

        if not norefresh:
            if crosshairs['t'] != self.t:
                self.set_frame(crosshairs['t'], noauto=True)
            for (x, y), ct in zip(self.crosshairs, self.crosshair_type):
                self._draw_crosshair((x, y), ct)
        else:
            return crosshairs['t']

    def make_button_panel(self, t: int = 0) -> widgets.Widget:
        # Break this out as a method so it can be overridden by subclasses.
        num_frames = min(len(loader) for loader in self.loaders)
        min_t, max_t = -min(self.shifts), num_frames - max(self.shifts)
        return ButtonPanel(
            self.set_frame, min_t, max_t, t, self.clear_crosshairs
        )

    def _remove_crosshair(self, pt: Tuple[int, int]):
        # Rescale to display coordinate system
        x = int(self.w * pt[0] / self.original_size[0])
        y = int(self.h * pt[1] / self.original_size[1])
        ch_w = max(int(self.sample_size * self.w / self.original_size[0]), 1)
        ch_h = max(int(self.sample_size * self.h / self.original_size[1]), 1)
        self.canvas[1].clear_rect(x - ch_w // 2, y - ch_h // 2, ch_w, ch_h)

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
            image = self.loaders[0].get_frame(
                t + self.shifts[0], for_display=True
            )

            if not noauto:
                self._autofind_crosshairs(image)
            self.set_images(image)

    def set_images(self, image: np.ndarray):
        if self.align_pipeline is not None:
            image = image.astype(np.float32) / 256.
            image = self.align_pipeline(image, np.empty_like(image))
            image = np.clip(256 * image, 0, 255).astype(np.uint8)

        if (image.ndim == 3) and (image.shape[2] == 3):
            image = image[:, :, ::-1]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        self.original_size = image.shape[:2][::-1]
        if (self.h, self.w) != image.shape[:2]:
            image = cv2.resize(image, (self.w, self.h))

        self.canvas[0].put_image_data(image)

    def _update_cache(self):
        if self.cache_path is not None:
            self.save_crosshairs(self.cache_path)
