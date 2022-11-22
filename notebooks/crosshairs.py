'''
This implements an :class:`ipywidgets.Widget` to perform alignment and
sampling tasks. This is distinct from the earlier version implemented in
matplotlib, and requires a Jupyter environment to run.
'''
import io
from typing import Dict, List, Optional, Tuple

import ipyevents as events
import ipywidgets as widgets
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traitlets

__all__ = ['ImageWithCrosshairs', 'SampleSelector']

ASSOCIATION_RADIUS_SQ = 16**2


def _resize(image: np.ndarray, out_size: Tuple[int, int]):
    # Convenience function for resizing images using PIL.
    return np.array(Image.fromarray(image).resize(out_size))


_PAIR = traitlets.Tuple(traitlets.Int(), traitlets.Int())

DEFAULT_CROSSHAIR = np.zeros((10, 10, 4), dtype=np.uint8)
DEFAULT_CROSSHAIR[:, :, 1] = DEFAULT_CROSSHAIR[:, :, 3] = 255
FONT = ImageFont.truetype('Inconsolata.otf', 32)
FONT_COLOR = (0, 255, 0)


@widgets.register
class ImageWithCrosshairs(widgets.Image):
    '''
    This extends :class:`NumpyImage` to add crosshairs to an image.
    '''
    # traitlets is a very cool library that sets up hooks to monitor setting
    # and updating attributes of a class. widgets.Image inherits from
    # traitlets.HasTraits. Traits are defined by setting class properties,
    # as follows:
    image = traitlets.Instance(klass=np.ndarray)
    # This sets image to always be a numpy.ndarray. The other trait we care
    # about is value, which we inherit frm widgets.Image and should contain the
    # encoding of the image array. As long as we ensure that value is the
    # correct encoding, and we have the class property format set to the
    # correct encoding - it defaults to 'png' - this will work like a
    # widgets.Image and be displayable in IPython.

    # crosshairs will be a list of pairs of integers storing current locations
    # of the crosshairs.
    crosshairs = traitlets.List(_PAIR)
    # These should be overridden by methods that handle a click on a crosshair
    # or a click on the background. handle_click_on_crosshair will be passed a
    # single int, corresponding to the index of the crosshair selected.
    # handle_crosshair_on_background will be passed a pair of ints,
    # corresponding to the x, y in pixel coordinates in the original image.
    handle_click_on_crosshair = None
    handle_click_on_background = None
    # This is used to avoid redrawing the image unnecessarily, by caching the
    # last set of crosshairs.
    cached_crosshairs = None

    def __init__(self, image: np.ndarray,
                 output_size: Optional[Tuple[int, int]] = None,
                 encoding: str = 'PNG',
                 crosshair_image: Optional[List[np.ndarray]] = None,
                 scale_crosshair: bool = False,
                 include_numbers: bool = False,
                 font: ImageFont.FreeTypeFont = FONT,
                 font_color: np.ndarray = FONT_COLOR):
        '''
        Args:
            image (:class:`numpy.ndarray`): The image array.
            output_size (optional, tuple of int): If this is provided, resize
            the image to this size in pixels before displaying.
            encoding (str): Encoding to use when sending the image to the
            browser.
            crosshair_image (optional, list of :class:`numpy.ndarray`):
            Optional, provides the image of the crosshair to be provided.
            Should be a list of 3-dimensional arrays in (H, W, C) arrangement,
            with four channels, where the 3rd channel is the alpha channel.
            scale_crosshair (bool): If true, scale the crosshair to preserve
            the ratio of its size to the *original* image's size.
            include_numbers (bool): If true, print index numbers next to the
            crosshairs to keep track of which is which.
            font (:class:`PIL.ImageFont.FreeTypeFont`): Font to use when
            printing numbers.
            font_color (:class:`numpy.ndarray`): Color to use when printing
            numbers.
        '''
        self.original_size = image.shape[:2][::-1]
        if output_size is not None:
            # Maintain aspect ratio when we resize. Note image.shape is (h, w),
            # output_size is (w, h). Our aspect ratios will be h / w.
            orig_ar = image.shape[0] / image.shape[1]
            new_ar = output_size[1] / output_size[0]
            if orig_ar >= new_ar:
                output_size = (output_size[0], output_size[1] * orig_ar)
            else:
                output_size = (output_size[0] / orig_ar, output_size[1])
            output_size = (int(output_size[0]), int(output_size[1]))
            image = _resize(image, output_size)

        # Coerce from monochrome to RGB
        if (image.ndim == 2) or (image.shape[2] == 1):
            image = image.reshape(*image.shape[:2], 1)
            image = np.concatenate([image] * 3, axis=2)

        # image_cache will store the resized, original image, so we can reuse
        # it if the crosshairs are updated.
        self.image_cache = image

        super().__init__(format=encoding)

        if crosshair_image is None:
            crosshair_image = [DEFAULT_CROSSHAIR]
        self._crosshair_image = crosshair_image
        self.scale_crosshair = scale_crosshair

        # Register method for handling clicks
        self.dom_handler = events.Event(source=self, watched_events=['click'])
        self.dom_handler.on_dom_event(self._handle_click)

        self.include_numbers = include_numbers
        self.font = font
        self.font_color = font_color
        # This assigns a unique integer ID to each point, that is preserved
        # even if a point is removed. It needs to be assigned before we call
        # super.__init__, as it will be expected by _observe_crosshairs.
        self.idxs: Optional[List[int]] = [] if include_numbers else None
        # This is used to track if shift was held down during the click. If
        # 0, the crosshair is self.crosshair_image. If 1, the crosshair is
        # self.alt_crosshair_image.
        self.crosshair_type: List[int] = []
        # Setting self.crosshairs also triggers _observe_crosshairs, which
        # ends up loading self.image_cache into self.image.
        self.crosshairs = []

    @staticmethod
    def _add_crosshairs(image: np.ndarray, crosshairs: List[Tuple[int, int]],
                        crosshair_image: List[np.ndarray],
                        crosshair_type: Optional[List[int]] = None,
                        original_size: Optional[Tuple[int]] = None,
                        numbers: Optional[List[int]] = None,
                        font: ImageFont.FreeTypeFont = FONT,
                        font_color: np.ndarray = FONT_COLOR):
        '''
        This add crosshairs to an image.
        '''
        # We're going to modify the image in place, so clone it.
        image = np.copy(image)
        # If original_size is not specified, assume the same as current size.
        h, w, *_ = image.shape
        original_size = original_size or (w, h)

        # Check if the image has been resized. The crosshair points have been
        # provided in the original coordinate space, and will need to be
        # updated.
        if original_size != (w, h):
            rs = (w / original_size[0], h / original_size[1])
            crosshairs = [(x * rs[0], y * rs[1]) for x, y in crosshairs]

        # Draw the crosshairs.
        masks, patches = [], []
        for xh in crosshair_image:
            h, w, *_ = xh.shape
            masks.append(xh[:, :, 3:4].astype(np.float64) / 255.)
            patches.append((xh[:, :, :3] * masks[-1]).astype(np.uint8))

        # crosshair_type defaults to all 0
        if crosshair_type is None:
            crosshair_type = [0] * len(crosshairs)

        for (x, y), s in zip(crosshairs, crosshair_type):
            ul_x, ul_y = max(int(x) - (w // 2), 0), max(int(y) - (h // 2), 0)
            lr_x, lr_y = ul_x + w, ul_y + h
            patch = (image[ul_y:lr_y, ul_x:lr_x, :] * (1 - masks[s]))
            image[ul_y:lr_y, ul_x:lr_x, :] = patch + patches[s]

        # Add the numbers, if desired.
        if numbers is not None:
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            for number, (x, y) in zip(numbers, crosshairs):
                x, y = int(x) + (w // 2) + 4, int(y) + (h // 2) + 4
                draw.text((x, y), str(number), font=font, fill=font_color)
            image = np.array(image)

        return image

    def _handle_click(self, event: Dict):
        '''
        This method handles click events and dispatches them to the correct
        method.
        '''
        # Extract point from event dictionary
        x, y = event['dataX'], event['dataY']
        # Get rescaling factor
        h, w, *_ = self.image.shape
        rs = (self.original_size[0] / w, self.original_size[1] / h)

        # If self.crosshairs is empty, the argmax will error out.
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
            self.min_idx = min_idx
            if callable(self.handle_click_on_crosshair):
                self.handle_click_on_crosshair(min_idx, event)
        elif callable(self.handle_click_on_background):
            # Rescale point to original coordinate system
            x, y = int(rs[0] * x), int(rs[1] * y)
            self.handle_click_on_background(x, y, event)

    # The trailets.observe('name') decorator registers a function that will
    # fire every time the attribute name changes.
    @traitlets.observe('crosshairs')
    def _observe_crosshairs(self, values: Dict):
        # values will be a dictionary. There are several keys, but the one we
        # care about is 'new', which contains the new value crosshairs is being
        # set to.
        self.image = self._add_crosshairs(
            image=self.image_cache,
            crosshairs=values['new'],
            crosshair_image=self.crosshair_image,
            crosshair_type=self.crosshair_type,
            original_size=self.original_size,
            numbers=self.idxs,
            font=self.font,
            font_color=self.font_color,
        )

    @traitlets.observe('image')
    def _observe_image(self, values: Dict):
        # values will be a dictionary. There are several keys, but the one we
        # care about is 'new', which contains the new value image is being set
        # to. This lets us automatically update value every time image is
        # updated. The encoding syntax is borrowed from PIL.Image._repr_png_.
        buff = io.BytesIO()
        try:
            Image.fromarray(values['new']).save(buff, self.format)
        except Exception:
            raise ValueError(f'Could not encode to {self.format}')
        else:
            self.value = buff.getvalue()

    @property
    def crosshair_image(self) -> List[np.ndarray]:
        if self.scale_crosshair:
            rs = self.image_cache.shape[1] / self.original_size[0]
            crosshair_images = []
            for image in self._crosshair_image:
                h, w, *_ = image.shape
                out_size = (int(w * rs), int(h * rs))
                crosshair_images.append(_resize(image, out_size))
            return crosshair_images

        else:
            return self._crosshair_image

    def reset(self):
        '''
        This clears all crosshairs from the image.
        '''
        with self.hold_trait_notifications():
            self.idxs = []
            self.crosshair_type = []
            self.crosshairs = []

    def set_image(self, image: np.ndarray):
        '''
        This changes the underlying image.
        '''
        self.original_size = image.shape[:2][::-1]
        self.image_cache = _resize(image, self.image.shape[:2][::-1])
        self.image = self._add_crosshairs(
            image=self.image_cache,
            crosshairs=self.crosshairs,
            crosshair_image=self.crosshair_image,
            crosshair_type=self.crosshair_type,
            original_size=self.original_size,
            numbers=self.idxs,
            font=self.font,
            font_color=self.font_color,
        )


@widgets.register
class SampleSelector(ImageWithCrosshairs):
    '''
    This combines :class:`ImageWithCrosshairs` with click-handling methods that
    allow selecting points on the image. To create a crosshair on the image,
    click on it. To remove that crosshair, click on it again.
    '''
    def get_points(self) -> np.ndarray:
        '''
        Returns the coordinates of the crosshairs.
        '''
        n_crosses = max(self.idxs) + 1
        pts = np.full((n_crosses, 2), float('nan'))
        pts[self.idxs] = np.array(self.crosshairs)[np.argsort(self.idxs)]
        return pts

    def handle_click_on_background(self, x: int, y: int, event: Dict):
        '''
        This fires if the user clicks on the background.

        Args:
            x (int): x coordinate of the click, in the original image
            coordinates.
            y (int): y coordinate of the click, in the original image
            coordinates.
        '''
        # Get the next index to be assigned by looking for the lowest index
        # that has not yet been used.
        if self.idxs:
            idx = min({i + 1 for i in self.idxs} - set(self.idxs))
        else:
            idx = 0
        self.idxs.append(idx)
        self.crosshair_type.append(int(event['shiftKey']))
        self.crosshairs.append((x, y))
        # We need to do this to get _observe_crosshairs to fire, because append
        # modifies in place.
        self._observe_crosshairs({'new': self.crosshairs})

    def handle_click_on_crosshair(self, idx: int, event: Dict):
        '''
        This fires if the user clicks on an existing crosshair.

        Args:
            idx (int): Index of the crosshair clicked on. This corresponds to
            the index in the crosshairs attribute, NOT the idxs list.
        '''
        # Remove the index.
        self.idxs.pop(idx)
        self.crosshair_type.pop(idx)
        self.crosshairs.pop(idx)
        # We need to do this to get _observe_crosshairs to fire, because pop
        # modifies in place.
        self._observe_crosshairs({'new': self.crosshairs})
