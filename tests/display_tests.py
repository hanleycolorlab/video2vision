from contextlib import contextmanager
import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
import video2vision as v2v

try:
    import tifffile
except ImportError:
    has_tiff = False
else:
    has_tiff = True


V2V_NB_ROOT = os.path.join(os.path.dirname(__file__), '../notebooks')
if V2V_NB_ROOT not in sys.path:
    sys.path.append(V2V_NB_ROOT)

import v2v_nb  # noqa

v2v_nb.displays.ASSOCIATION_RADIUS_SQ = 0.25


class MockCanvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data = np.zeros((height, width, 4), dtype=np.uint8)

    def clear_rect(self, x: int, y: int, w: int, h: int):
        self.data[y:y + h + 1, x:x + w + 1, :] = 0

    def put_image_data(self, image: np.ndarray, x: int = 0, y: int = 0):
        ul_x, ul_y = x, y
        lr_x, lr_y = ul_x + image.shape[1], ul_y + image.shape[0]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=2)
        if image.shape[2] == 3:
            image = np.pad(
                image, ((0, 0), (0, 0), (0, 1)), constant_values=255
            )
        self.data[ul_y:lr_y, ul_x:lr_x, :] = image

    def stroke_line(self, x_1: int, y_1: int, x_2: int, y_2: int):
        color = (
            int(self.stroke_style[1:3], 16),
            int(self.stroke_style[3:5], 16),
            int(self.stroke_style[5:7], 16),
            255
        )
        cv2.line(self.data, (x_1, y_1), (x_2, y_2), color)

    def stroke_rect(self, x: int, y: int, w: int, h: int):
        self.stroke_line(x, y, x + w, y)
        self.stroke_line(x + w, y, x + w, y + h)
        self.stroke_line(x + w, y + h, x, y + h)
        self.stroke_line(x, y + h, x, y)


class MockMultiCanvas:
    def __init__(self, canvases):
        self.canvases = canvases

    def __getitem__(self, i: int) -> MockCanvas:
        return self.canvases[i]

    @classmethod
    def substitute(cls, display_box):
        canvas = cls([
            MockCanvas(width=display_box.w, height=display_box.h),
            MockCanvas(width=display_box.w, height=display_box.h),
        ])
        for i in [0, 1]:
            canvas[i].stroke_style = display_box.canvas[i].stroke_style
        display_box.canvas = canvas
        display_box.set_frame(display_box.t)

    def get_image_data(self) -> np.ndarray:
        s = self.canvases[1].data[:, :, 3:4] / 255.
        rgb = (
            (self.canvases[0].data[:, :, :3] * (1 - s)) +
            (self.canvases[1].data[:, :, :3] * s)
        )
        a = np.maximum(
            self.canvases[0].data[:, :, 3:4],
            self.canvases[1].data[:, :, 3:4],
        )
        return np.concatenate((rgb.astype(np.uint8), a), axis=2)


class ActiveDisplayTest(unittest.TestCase):
    @contextmanager
    def with_images(self, sep: bool = False, rgb: bool = True,
                    ext: str = 'png', dtype=np.uint8):
        with tempfile.TemporaryDirectory() as temp_root:
            image = np.zeros((8, 8, 3) if rgb else (8, 8), dtype=dtype)
            image[0, 0] = 255

            path_0 = os.path.join(temp_root, f'0.{ext}')
            if ext == 'tif':
                self.assertTrue(has_tiff, 'Needs tifffile for this test')
                tifffile.imwrite(
                    path_0, np.moveaxis(image, -1, 0), photometric='minisblack'
                )
            else:
                Image.fromarray(image).save(path_0)

            image[:] = 0
            path_1 = os.path.join(temp_root, f'1.{ext}')
            if ext == 'tif':
                self.assertTrue(has_tiff, 'Needs tifffile for this test')
                tifffile.imwrite(
                    path_1, np.moveaxis(image, -1, 0), photometric='minisblack'
                )
            else:
                Image.fromarray(image).save(path_1)

            if sep:
                yield (
                    v2v.Loader(path_0, (8, 8), num_channels=(3 if rgb else 1)),
                    v2v.Loader(path_1, (8, 8), num_channels=(3 if rgb else 1)),
                )
            else:
                yield v2v.Loader(
                    os.path.join(temp_root, f'*.{ext}'), (8, 8),
                    num_channels=(3 if rgb else 1),
                )

    def test_display_box(self):
        with self.with_images() as loader:
            display_box = v2v_nb.DisplayBox(
                loader, t=0, shifts=(1,), output_size=(4, 4),
            )
            MockMultiCanvas.substitute(display_box)

            # TODO: Hook display_image instead of display
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            mask = np.ones((4, 4, 4), dtype=bool)
            mask[0, 0, :] = mask[:, :, 3] = False

            display_box.buttons.children[2].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[0, 0, 3] == 255).all())
            self.assertTrue((display_image[mask] == 0).all())

            display_box.buttons.children[0].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[0, 0, 3] == 255).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_display_box_monochrome(self):
        with self.with_images(rgb=False) as loader:
            display_box = v2v_nb.DisplayBox(
                loader, t=0, shifts=(1,), output_size=(4, 4),
            )
            MockMultiCanvas.substitute(display_box)

            # TODO: Hook display_image instead of display
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            mask = np.ones((4, 4), dtype=bool)
            mask[0, 0] = False

            display_box.buttons.children[2].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[mask, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            display_box.buttons.children[0].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[mask, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

    def test_display_box_tiff(self):
        with self.with_images(ext='tif', dtype=np.float32) as loader:
            display_box = v2v_nb.DisplayBox(
                loader, t=0, shifts=(1,), output_size=(4, 4),
            )
            MockMultiCanvas.substitute(display_box)

            # TODO: Hook display_image instead of display
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            mask = np.ones((4, 4), dtype=bool)
            mask[0, 0] = False

            display_box.buttons.children[2].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[mask, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            display_box.buttons.children[0].click()
            display_image = display_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == 64).all())
            self.assertTrue((display_image[mask, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

    def test_ghost_box(self):
        with self.with_images(True) as (loader_0, loader_1):
            ghost_box = v2v_nb.GhostBox(
                loader_0, loader_1, output_size=(4, 4),
            )
            MockMultiCanvas.substitute(ghost_box)

            mask = np.ones((4, 4, 4), dtype=bool)
            mask[0, 0, :] = mask[:, :, 3] = False

            # TODO: Hook display_image instead of display
            display_image = ghost_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 0, :3] == (64, 0, 64)).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_selector_box(self):
        with self.with_images() as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )
            MockMultiCanvas.substitute(selector_box)

            # TODO: Hook display_image instead of display
            display_image = selector_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 1,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click(
                {'relativeX': 2, 'relativeY': 2, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[2:4, 2:4, 1] = 255
            should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[2:4, 2:4, 1] = should_be[:, :, 3] = 255
            should_be[2, 2, 1] = 0
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            with self.assertRaises(RuntimeError):
                selector_box.get_samples()

            # Get samples
            selector_box.idxs = [0]
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (1, 3))
            self.assertTrue((sample_values == 0).all())
            self.assertEqual(sample_types.shape, (1,))
            self.assertTrue((sample_types == 0).all())

            # Clear all
            selector_box.children[1].children[-1].click()
            self.assertEqual(selector_box.idxs, [])
            self.assertEqual(selector_box.crosshairs, [])
            self.assertEqual(selector_box.crosshair_type, [])
            display_image = selector_box.canvas.get_image_data()
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[:, :, 3] = 255
            self.assertTrue((display_image == should_be).all())

            # Check it doesn't throw an error if there's no samples selected
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (0, 3))
            self.assertEqual(sample_types.shape, (0,))

    def test_selector_box_with_align(self):
        align_pipe = v2v.Pipeline()
        loader_idx = align_pipe.add_operator(
            v2v.Loader(None, expected_size=(8, 8))
        )
        align_pipe.add_operator(v2v.Loader(None, expected_size=(8, 8)))
        flip_idx = align_pipe.add_operator(v2v.HorizontalFlip())
        writer_idx = align_pipe.add_operator(v2v.Writer(extension='png'))
        align_pipe.add_edge(loader_idx, flip_idx, in_slot=0)
        align_pipe.add_edge(flip_idx, writer_idx, in_slot=0)

        with self.with_images() as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, output_size=(4, 4), align_pipeline=align_pipe, w=1,
            )
            MockMultiCanvas.substitute(selector_box)
            display_image = selector_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))

            mask = np.ones((4, 4, 3), dtype=bool)
            mask[0, 3, :] = False

            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[0, 3, :3] == 64).all())
            self.assertTrue((display_image[:, :, :3][mask] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            selector_box.crosshairs = [(7, 0), (0, 2)]
            selector_box.crosshair_type = [True, True]
            selector_box.idxs = [0, 1]

            samples, _ = selector_box.get_samples()
            should_be = np.array([[1., 1., 1.], [0., 0., 0.]])
            self.assertTrue((np.abs(samples - should_be) < 0.01).all())

    def test_selector_box_with_monochrome(self):
        with self.with_images(rgb=False) as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )
            MockMultiCanvas.substitute(selector_box)

            # TODO: Hook display_image instead of display
            display_image = selector_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 1,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click(
                {'relativeX': 2, 'relativeY': 2, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[2:4, 2:4, 1] = 255
            should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[2:4, 2:4, 1] = should_be[:, :, 3] = 255
            should_be[2, 2, 1] = 0
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            with self.assertRaises(RuntimeError):
                selector_box.get_samples()

            # Get samples
            selector_box.idxs = [0]
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (1, 1))
            self.assertTrue((sample_values == 0).all())
            self.assertEqual(sample_types.shape, (1,))
            self.assertTrue((sample_types == 0).all())

            # Clear all
            selector_box.children[1].children[-1].click()
            self.assertEqual(selector_box.idxs, [])
            self.assertEqual(selector_box.crosshairs, [])
            self.assertEqual(selector_box.crosshair_type, [])
            display_image = selector_box.canvas.get_image_data()
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[:, :, 3] = 255
            self.assertTrue((display_image == should_be).all())

            # Check it doesn't throw an error if there's no samples selected
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (0, 3))
            self.assertEqual(sample_types.shape, (0,))

    def test_selector_box_with_tif(self):
        with self.with_images(ext='tif', dtype=np.float32) as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )
            MockMultiCanvas.substitute(selector_box)

            # TODO: Hook display_image instead of display
            display_image = selector_box.canvas.get_image_data()
            self.assertEqual(display_image.shape, (4, 4, 4))
            self.assertTrue((display_image[:, :, :3] == 0).all())
            self.assertTrue((display_image[:, :, 3] == 255).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 1,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click(
                {'relativeX': 2, 'relativeY': 2, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[1:3, 1:3, 1] = should_be[2:4, 2:4, 1] = 255
            should_be[:, :, 3] = 255
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click(
                {'relativeX': 1, 'relativeY': 1, 'shiftKey': 0,
                 'boundingRectWidth': 4, 'boundingRectHeight': 4}
            )
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[2:4, 2:4, 1] = should_be[:, :, 3] = 255
            should_be[2, 2, 1] = 0
            display_image = selector_box.canvas.get_image_data()
            self.assertTrue((display_image == should_be).all())

            with self.assertRaises(RuntimeError):
                selector_box.get_samples()

            # Get samples
            selector_box.idxs = [0]
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (1, 3))
            self.assertTrue((sample_values == 0).all())
            self.assertEqual(sample_types.shape, (1,))
            self.assertTrue((sample_types == 0).all())

            # Clear all
            selector_box.children[1].children[-1].click()
            self.assertEqual(selector_box.idxs, [])
            self.assertEqual(selector_box.crosshairs, [])
            self.assertEqual(selector_box.crosshair_type, [])
            display_image = selector_box.canvas.get_image_data()
            should_be = np.zeros((4, 4, 4), dtype=np.uint8)
            should_be[:, :, 3] = 255
            self.assertTrue((display_image == should_be).all())

            # Check it doesn't throw an error if there's no samples selected
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (0, 3))
            self.assertEqual(sample_types.shape, (0,))

    def test_selector_box_offside_crosshair(self):
        crosshairs = {
            't': 0,
            'idxs': [0, 1],
            'crosshair_type': [0, 0],
            'crosshairs': [(0, 0), (8, 8)],
        }

        with tempfile.TemporaryDirectory() as temp_root:
            crosshair_path = os.path.join(temp_root, 'crosshairs.json')
            with open(crosshair_path, 'w') as crosshair_file:
                json.dump(crosshairs, crosshair_file)

            with self.with_images() as loader:
                selector_box = v2v_nb.SelectorBox(
                    loader, output_size=(8, 8), marker_choice='box', w=4,
                )
                MockMultiCanvas.substitute(selector_box)
                selector_box.load_crosshairs(crosshair_path)
                display_image = selector_box.canvas.get_image_data()

        self.assertEqual(display_image.shape, (8, 8, 4))
        should_be = np.array([
            [  0,   0, 255,   0,   0,   0,   0,   0],
            [  0,   0, 255,   0,   0,   0,   0,   0],
            [255, 255, 255,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0, 255, 255],
            [  0,   0,   0,   0,   0,   0, 255,   0],
        ])
        should_be = np.stack(
            [np.zeros_like(should_be), should_be, np.zeros_like(should_be),
             np.full_like(should_be, 255)],
            axis=2
        )
        should_be[0, 0, :] = 255
        self.assertTrue((display_image == should_be).all())


if __name__ == '__main__':
    unittest.main()
