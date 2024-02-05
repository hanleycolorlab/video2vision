from contextlib import contextmanager
import json
from typing import Callable, List, Optional, Tuple, Union

import cv2
import ipywidgets as widgets
import numpy as np
from PIL import Image
import video2vision as v2v

from crosshairs import SampleSelector

__all__ = ['DisplayBox', 'ForwardBackwardButtons', 'SelectorBox']


class DisplayBox(widgets.VBox):
    '''
    This is a widget for displaying images from two or more different
    :class:`v2v.Loader`s side by side. It provides buttons for paging forward
    or backward in a video or set of images, but no other functionality.
    '''
    def __init__(self, *loaders, t: int = 0,
                 shifts: Optional[List[int]] = None,
                 preprocess: Optional[Callable] = None):
        '''
        Args:
            loaders (:class:`video2vision.Loader`): The loaders to use.
            t (int): Initial time value.
            shifts (optional, list of int): If provided, this gives offsets to
            apply to the loaders relative to each other, e.g. if there is a
            temporal shift between the sources.
            preprocess (optional, callable): If provided, this function is
            called on the inputs prior to displaying them.
        '''
        self.loaders, self.preprocess = loaders, preprocess
        if shifts is None:
            shifts = [0 for _ in self.loaders]
        self.shifts = shifts

        images = [r.get_frame(t + s) for r, s in zip(self.loaders, shifts)]
        if self.preprocess is not None:
            test_images = self.preprocess(*images)
        else:
            test_images = images
        h = max(image.shape[0] for image in test_images)
        w = sum(image.shape[1] for image in test_images)
        num_frames = min(len(loader) for loader in self.loaders)
        min_t, max_t = -min(shifts), num_frames - max(shifts)

        self.display = Image.new('RGB', (w, h))
        self.buttons = ForwardBackwardButtons(self.set_frame, min_t, max_t, t)
        self.display_image = widgets.Image(format='png')
        self.set_images(*images)

        super().__init__((self.display_image, self.buttons))

    def set_frame(self, t: int):
        '''
        Sets the current frame.
        '''
        with self.buttons.disable():
            images = [
                r.get_frame(t + s) for r, s in zip(self.loaders, self.shifts)
            ]
            self.set_images(*images)

    def set_images(self, *images):
        if self.preprocess is not None:
            images = self.preprocess(*images)
        # Apply gamma scaling to ensure visually correct display
        images = [gamma_scale(image) for image in images]

        x = 0
        for image in images:
            # Convert from BGR, float32, [0, 1] to RGB, uint8, [0, 255]
            if (image.ndim == 3) and (image.shape[2] == 3):
                image = image[:, :, ::-1]
            image = np.clip(256 * image, 0, 255).astype(np.uint8)
            self.display.paste(Image.fromarray(image), (x, 0))
            x += image.shape[1]
        self.display_image.value = self.display._repr_png_()

    @property
    def t(self) -> int:
        return self.buttons.t


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


class GhostBox(widgets.VBox):
    '''
    This widget overlays two images on each other, using one to provide the
    red and blue channels and the other to provide the green channel. This is
    used to test the alignment of two images.
    '''
    def __init__(self, loader_0, loader_1, t: int = 0,
                 shifts: Optional[List[int]] = None):
        self.loader_0, self.loader_1 = loader_0, loader_1
        if shifts is None:
            shifts = [0, 0]
        self.shifts = shifts

        image_0 = self.loader_0.get_frame(t + shifts[0])
        image_1 = self.loader_1.get_frame(t + shifts[1])

        num_frames = min(len(r) for r in self.loaders)
        min_t, max_t = -min(shifts), num_frames - max(shifts)

        self.buttons = ForwardBackwardButtons(self.set_frame, min_t, max_t, t)
        self.display_image = widgets.Image(format='png')
        self.set_images(image_0, image_1)

        super().__init__((self.display_image, self.buttons))

    @property
    def loaders(self):
        return self.loader_0, self.loader_1

    def set_frame(self, t: int):
        with self.buttons.disable():
            images = [
                r.get_frame(t + s) for r, s in zip(self.loaders, self.shifts)
            ]
            self.set_images(*images)

    def set_images(self, image_0, image_1):
        image_0[..., 1] = image_1[..., 1]
        # Apply gamma scaling to ensure visually correct display
        image_0 = gamma_scale(image_0)
        image = np.clip(256 * image_0[..., ::-1], 0, 255).astype(np.uint8)
        self.display_image.value = Image.fromarray(image)._repr_png_()

    @property
    def t(self) -> int:
        return self.buttons.t


class SelectorBox(widgets.VBox):
    '''
    This widget allows the user to select points on an image. It is intended to
    be used to select sample points.
    '''
    def __init__(self, loader: v2v.Loader, t: int = 0, w: int = 25,
                 preprocess: Optional[Callable] = None,
                 box_color: np.ndarray = (0, 255, 0),
                 font_color: Optional[np.ndarray] = None,
                 auto_op: Optional[Union[str, v2v.AutoLinearize]] = None):
        '''
        Args:
            loader (:class:`video2vision.Loader`): Loader pointing to source of
            images.
            t (int): Initial time.
            w (int): Width of selector box in pixels.
            preprocess (optional, callable): If provided, apply this function
            to the image prior to using it.
            box_color (:class:`numpy.ndarray`): Color of the selector boxes.
            font_color (optional, :class:`numpy.ndarray`): Color of the numbers
            next to the sample boxes. If not provided, equal to box_color.
            auto_op (optional, str or :class:`video2vision.AutoLinearize`): If
            provided, use this autolinearizer to automatically locate sample
            locations. If this is a string, assume it is the path to a saved
            :class:`video2vision.AutoLinearize`.
        '''
        self.preprocess, self.loader = preprocess, loader
        font_color = box_color if (font_color is None) else font_color
        box_color = np.concatenate((np.array(box_color), (255,)))
        self.sample_size = w
        crosshair_image = np.zeros((w, w, 4), dtype=np.uint8)
        r = max(1, w // 25)
        crosshair_image[:r, :, :] = crosshair_image[-r:, :, :] = box_color
        crosshair_image[:, :r, :] = crosshair_image[:, -r:, :] = box_color
        shift_image = np.zeros((w, w, 4), dtype=np.uint8)
        shift_image[np.arange(w), np.arange(w), :] = box_color
        shift_image[np.arange(w), -np.arange(1, w + 1), :] = box_color
        if isinstance(auto_op, str):
            self.auto_op = v2v.load_operator(auto_op)
        else:
            self.auto_op = None
        image = self.loader.get_frame(t)
        # The order we add these attributes here is important for
        # GuidedSelectorBox, so that the self.t property will work.
        self.buttons = ForwardBackwardButtons(
            self.set_frame, 0, len(self.loader), t
        )
        self.selector = SampleSelector(
            self._prep(image),
            (1000, 1000),
            crosshair_image=[crosshair_image, shift_image],
            scale_crosshair=True,
            include_numbers=True,
            font_color=font_color,
            border_margin=max(w // 2, 1),
        )
        super().__init__((self.selector, self.buttons))
        self._autofind_crosshairs()

    def _autofind_crosshairs(self):
        if self.auto_op is not None:
            try:
                crosshairs, _ = self.auto_op._locate_samples(
                    self._cached_image
                )
            except Exception:
                pass
            else:
                self.selector.idxs = list(range(crosshairs.shape[0]))
                self.selector.crosshair_type = [0 for _ in self.selector.idxs]
                self.selector.crosshairs = crosshairs.astype(np.int64).tolist()

    @property
    def crosshairs(self) -> List[Tuple[int, int]]:
        return self.selector.crosshairs

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        if max(self.selector.idxs) + 1 != len(self.selector.idxs):
            raise ValueError('Not all samples selected')
        samples = v2v.utils.extract_samples(
            self._cached_image, self.crosshairs, self.sample_size,
        )
        types = np.array(self.selector.crosshair_type, dtype=bool)
        # Reindex samples
        reidx = [self.selector.idxs.index(i) for i in range(samples.shape[0])]
        return samples[reidx], types[reidx]

    @property
    def image(self) -> np.ndarray:
        return self.children[0].image_cache

    def load_crosshairs(self, path: str):
        with open(path, 'r') as crosshairs_file:
            crosshairs = json.load(crosshairs_file)
        self.set_frame(crosshairs['t'])
        self.buttons.t = crosshairs['t']
        self.selector.idxs = crosshairs['idxs']
        self.selector.crosshair_type = crosshairs['crosshair_type']
        self.selector.crosshairs = crosshairs['crosshairs']

    def _prep(self, image: np.ndarray) -> np.ndarray:
        if self.preprocess is not None:
            image = self.preprocess(image)
        self._cached_image = image
        # Apply gamma scaling to ensure visually correct display
        image = gamma_scale(image)

        # Loader returns in BGR, but SampleSelector expects RGB. Also, we need
        # to convert float32, [0, 1] to uint8, [0, 255]
        if (image.ndim == 3) and (image.shape[2] == 3):
            image = image[:, :, ::-1]
        # Image.fromarray doesn't like 3-dimensional monochrome images.
        elif (image.ndim == 3) and (image.shape[2] == 1):
            image = image.reshape(*image.shape[:2])
        return np.clip(256 * image, 0, 255).astype(np.uint8)

    def save_crosshairs(self, path: str):
        crosshairs = {
            'idxs': self.selector.idxs,
            'crosshair_type': self.selector.crosshair_type,
            'crosshairs': self.selector.crosshairs,
            't': self.t,
        }

        with open(path, 'w') as crosshairs_file:
            json.dump(crosshairs, crosshairs_file)

    def set_frame(self, t: int):
        with self.buttons.disable(), self.selector.hold_trait_notifications():
            self.selector.reset()
            image = self.loader.get_frame(t)
            self.selector.set_image(self._prep(image))
            self._autofind_crosshairs()

    @property
    def t(self) -> int:
        return self.buttons.t


class GuidedSelectorBox(SelectorBox):
    def __init__(self, loader: v2v.Loader, coloader: v2v.Loader,
                 expected_values: np.ndarray, percentile: float = 1.,
                 t: int = 0, w: int = 25,
                 preprocess: Optional[Callable] = None,
                 joint_preprocess: Optional[Callable] = None,
                 contour_color: np.ndarray = (0, 255, 0),
                 box_color: np.ndarray = (0, 255, 0),
                 font_color: Optional[np.ndarray] = None,
                 auto_op: Optional[Union[str, v2v.AutoLinearize]] = None):
        self.coloader = coloader
        self.joint_preprocess = joint_preprocess
        norms = np.linalg.norm(expected_values, axis=1, keepdims=True)
        self.expected_values = expected_values / norms
        self.percentile = percentile
        # Normalize to [0, 1] instead of [0, 256].
        self.contour_color = np.array(contour_color) / 255.

        super().__init__(
            loader, t=t, w=w, preprocess=preprocess, box_color=box_color,
            font_color=font_color, auto_op=auto_op,
        )

    def _prep(self, image: np.ndarray) -> np.ndarray:
        if self.preprocess is not None:
            image = self.preprocess(image)
        # Some ops below will modify image in-place, and we don't want that to
        # effect the cached copy.
        self._cached_image = image.copy()

        if self.joint_preprocess is not None:
            coimage = self.coloader.get_frame(self.t)
            guide = self.joint_preprocess(coimage, image)
        else:
            guide = image

        # Normalize to constant magnitude
        guide /= np.linalg.norm(guide, axis=2, keepdims=True)

        # Number of bands
        n_b = guide.shape[2]

        # We do this one row at a time to reduce memory requirements.
        for i, row in enumerate(self.expected_values):
            # Find the regions that are within the specified percentile of the
            # target value.
            d = np.dot(guide.reshape(-1, n_b), row.reshape(n_b, 1))
            d = np.arccos(d).reshape(*guide.shape[:2])
            thresh = np.percentile(d.flatten(), self.percentile)
            mask = (d <= thresh).astype(np.uint8)
            # Get contours out of the mask.
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
            )
            # Drop contours with small area.
            areas = [cv2.contourArea(c) for c in contours]
            thresh = np.percentile(areas, 99)
            contours = [c for c, a in zip(contours, areas) if a > thresh]
            # Draw the contours on the image.
            for contour in contours:
                cv2.drawContours(image, [contour], 0, self.contour_color, 3)
                # Add text label
                pt = contour.reshape(-1, 2).min(0)
                if (pt < 0).any():
                    pt = contour.reshape(-1, 2).max(0)
                cv2.putText(
                    image, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 3,
                    self.contour_color, 2, cv2.LINE_AA,
                )

        # Loader returns in BGR, but SampleSelector expects RGB. Also, we need
        # to convert float32, [0, 1] to uint8, [0, 255]
        if (image.ndim == 3) and (image.shape[2] == 3):
            image = image[:, :, ::-1]
        # Image.fromarray doesn't like 3-dimensional monochrome images.
        elif (image.ndim == 3) and (image.shape[2] == 1):
            image = image.reshape(*image.shape[:2])
        return np.clip(256 * image, 0, 255).astype(np.uint8)


def gamma_scale(image):
    # Coerce to [0, 1] range
    pix_min = image.min((0, 1), keepdims=True)
    pix_max = image.max((0, 1), keepdims=True)
    image = (image - pix_min) / (pix_max - pix_min)
    return image ** 2.2
