from contextlib import contextmanager, redirect_stdout
import csv
import io
import json
from math import isclose
import os
import shutil
import sys
import tempfile
from typing import Any, Optional, Type, Union
import unittest

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


def _create_dummy_csv(path: str, value: Union[np.ndarray, float]):
    if isinstance(value, (float, int)):
        value = np.full((401, 4), value)
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    if value.ndim == 1:
        value = np.stack([value] * 401, axis=0)

    with open(path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['wl'] + [str(i) for i in range(value.shape[1])])
        for i in range(401):
            writer.writerow([i + 300] + value[i].tolist())


class ChoicesTest(unittest.TestCase):
    def _test_widget(self, key: str, cls: Type, v_1: Any, v_2: Any,
                     none_value: Any = ''):
        config = v2v_nb.get_config()
        config[key] = v_1

        widget = cls(key)
        self.assertEqual(widget.children[0].value, 'You should never see this')
        self.assertTrue(config._nb_labels[key] is widget.favorite)
        self.assertEqual(widget.value, v_1)

        widget.value = v_2
        self.assertEqual(config[key], v_2)

        widget_2 = cls(key)
        self.assertTrue(widget_2.favorite is widget.favorite)

        config[key] = v_1
        self.assertEqual(widget.value, v_1)

        config[key] = None
        self.assertEqual(widget.value, none_value)

    def test_array_box(self):
        config = v2v_nb.get_config()
        config['test_array'] = None

        widget = v2v_nb.ArrayBox('test_array')
        self.assertEqual(widget.children[0].value, 'You should never see this')
        self.assertTrue(config._nb_labels['test_array'] is widget.favorite)
        for box in widget.favorite:
            self.assertEqual(box.value, '')
        self.assertEqual(widget.value, None)

        config['test_array'] = np.ones((3, 3))
        for box in widget.favorite:
            self.assertEqual(float(box.value), 1.)
        self.assertTrue(isinstance(widget.value, np.ndarray))
        self.assertEqual(widget.value.shape, (3, 3))
        self.assertTrue((widget.value == 1.).all())

        config['test_array'] = None
        for box in widget.favorite:
            self.assertEqual(box.value, '')
        self.assertEqual(widget.value, None)

        config['test_array'] = np.ones((3, 3))
        widget.children[1].children[1].click()
        for box in widget.favorite:
            self.assertEqual(box.value, '')
        self.assertEqual(widget.value, None)

    def test_bool_box(self):
        self._test_widget('test_bool', v2v_nb.BoolBox, True, False, False)

    def test_int_box(self):
        config = v2v_nb.get_config()
        config['test_int'] = 1

        widget = v2v_nb.IntBox('test_int')
        self.assertEqual(widget.children[0].value, 'You should never see this')
        self.assertTrue(config._nb_labels['test_int'] is widget.favorite)
        self.assertEqual(widget.value, '1')

        widget.value = '2'
        self.assertEqual(config['test_int'], 2)

        widget_2 = v2v_nb.IntBox('test_int')
        self.assertTrue(widget_2.favorite is widget.favorite)

        config['test_int'] = 1
        self.assertEqual(widget.value, '1')

    def test_path_box(self):
        self._test_widget('test_path', v2v_nb.PathBox, '/tmp', '/unq')

    def test_simple_button(self):
        self._test_value = False

        def action(*x):
            self._test_value = True

        button = v2v_nb.SimpleButton('', action)

        button.click()

        self.assertTrue(self._test_value)

        del self._test_value

    def test_string_box(self):
        self._test_widget('test_str', v2v_nb.StringBox, 'asdasd', 'efgefg')


class ConfigTest(unittest.TestCase):
    @contextmanager
    def make_pipe(self):
        with tempfile.TemporaryDirectory() as temp_root:
            pipe = v2v.Pipeline()
            in_path = os.path.join(temp_root, 'asd.png')
            pipe.add_operator(v2v.Loader(in_path, expected_size=(8, 10)))
            out_path = os.path.join(temp_root, 'asd.png')
            pipe.add_operator(v2v.Writer(out_path, extension='png'))
            temp_path = os.path.join(temp_root, 'pipe.json')
            pipe.save(temp_path)
            yield temp_path

    def test_cache(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        config['is_sony_camera'] = True

        with tempfile.TemporaryDirectory() as temp_root:
            # Should save at this point
            config['experiment_name'] = temp_root

            config['experiment_name'] = None
            config['is_sony_camera'] = False

            # This should reload the cache
            config['experiment_name'] = temp_root
            self.assertTrue(config['is_sony_camera'])

    def test_image_size_and_out_extension(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.assertRaises(v2v_nb.ParamNotSet):
            config.image_size
        with self.assertRaises(v2v_nb.ParamNotSet):
            config.out_extension

        with self.make_pipe() as pipe_path:
            config['align_pipe_path'] = pipe_path
            self.assertEqual(config.image_size, (8, 10))

        # Note this is NOT inside the context manager. This tests that BOTH
        # properties are pulled in, since the pipe.json no longer exists at
        # this point.
        self.assertEqual(config.out_extension, 'png')

    def test_out_path(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.assertRaises(v2v_nb.ParamNotSet):
            config.out_path

        with self.make_pipe() as pipe_path:
            config['align_pipe_path'] = pipe_path

            for k in ['uv_path', 'vis_path', 'uv_aligned_path']:
                with self.assertRaises(v2v_nb.ParamNotSet):
                    config.out_path
                config[k] = 'asd.png'

            self.assertEqual(config.out_path, 'asd.png')

            config[k] = 'edf'
            self.assertEqual(config.out_path, 'edf/asd-asd.png')

    def test_caching(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with tempfile.TemporaryDirectory() as temp_root:
            config['experiment_name'] = temp_root
            config['test_str'] = 'abcdef'
            config['experiment_name'] = None
            config['test_str'] = 'ghijkl'
            config['experiment_name'] = temp_root
            self.assertEqual(config['test_str'], 'abcdef')

        config['experiment_name'] = None


class DisplayTest(unittest.TestCase):
    @contextmanager
    def with_images(self, sep: bool = False, rgb: bool = True,
                    ext: str = 'png', dtype=np.uint8):
        with tempfile.TemporaryDirectory() as temp_root:
            image = np.zeros((8, 8, 3) if rgb else (8, 8), dtype=dtype)
            image[0, 0] = 255
            path_0 = os.path.join(temp_root, f'0.{ext}')
            if ext == 'tif':
                self.assertTrue(has_tiff, 'Needs tifffile for this test')
                tifffile.imwrite(path_0, image, photometric='minisblack')
            else:
                Image.fromarray(image).save(path_0)

            image[:] = 0
            path_1 = os.path.join(temp_root, f'1.{ext}')
            if ext == 'tif':
                self.assertTrue(has_tiff, 'Needs tifffile for this test')
                tifffile.imwrite(path_1, image, photometric='minisblack')
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

            # TODO: Hook display_image instead of display
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image == 0).all())

            mask = np.ones((4, 4, 3), dtype=bool)
            mask[0, 0, :] = False

            display_box.buttons.children[2].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

            display_box.buttons.children[0].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_display_box_monochrome(self):
        with self.with_images(rgb=False) as loader:
            display_box = v2v_nb.DisplayBox(
                loader, t=0, shifts=(1,), output_size=(4, 4),
            )

            # TODO: Hook display_image instead of display
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image == 0).all())

            mask = np.ones((4, 4), dtype=bool)
            mask[0, 0] = False

            display_box.buttons.children[2].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

            display_box.buttons.children[0].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_display_box_tiff(self):
        with self.with_images(ext='tif', dtype=np.float32) as loader:
            display_box = v2v_nb.DisplayBox(
                loader, t=0, shifts=(1,), output_size=(4, 4),
            )

            # TODO: Hook display_image instead of display
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image == 0).all())

            mask = np.ones((4, 4), dtype=bool)
            mask[0, 0] = False

            display_box.buttons.children[2].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

            display_box.buttons.children[0].click()
            display_image = np.array(display_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_ghost_box(self):
        with self.with_images(True) as (loader_0, loader_1):
            ghost_box = v2v_nb.GhostBox(loader_0, loader_1, output_size=(4, 4))

            mask = np.ones((4, 4, 3), dtype=bool)
            mask[0, 0, :] = False

            # TODO: Hook display_image instead of display
            display_image = np.array(ghost_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 0] == (64, 0, 64)).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_selector_box(self):
        with self.with_images() as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )

            # TODO: Hook display_image instead of display
            display_image = np.array(selector_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image == 0).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 1})
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[1, 1, 1] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click({'dataX': 2, 'dataY': 2, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[1, 1, 1] = should_be[2, 2, 1] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[2, 2, 1] = 255
            display_image = np.array(selector_box.display)
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
            display_image = np.array(selector_box.display)
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
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
                loader, output_size=(4, 4), align_pipeline=align_pipe,
            )
            display_image = np.array(selector_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))

            mask = np.ones((4, 4, 3), dtype=bool)
            mask[0, 3, :] = False

            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image[0, 3] == 64).all())
            self.assertTrue((display_image[mask] == 0).all())

    def test_selector_box_with_monochrome(self):
        with self.with_images(rgb=False) as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )

            # TODO: Hook display_image instead of display
            display_image = np.array(selector_box.display)
            self.assertEqual(display_image.shape, (4, 4))
            self.assertTrue((display_image == 0).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 1})
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4), dtype=np.uint8)
            should_be[1, 1] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click({'dataX': 2, 'dataY': 2, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4), dtype=np.uint8)
            should_be[1, 1] = should_be[2, 2] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4), dtype=np.uint8)
            should_be[2, 2] = 255
            display_image = np.array(selector_box.display)
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
            display_image = np.array(selector_box.display)
            should_be = np.zeros((4, 4), dtype=np.uint8)
            self.assertTrue((display_image == should_be).all())

            # Check it doesn't throw an error if there's no samples selected
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (0, 1))
            self.assertEqual(sample_types.shape, (0,))

    def test_selector_box_with_tif(self):
        with self.with_images(ext='tif', dtype=np.float32) as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(4, 4), t=1,
            )

            # TODO: Hook display_image instead of display
            display_image = np.array(selector_box.display)
            self.assertEqual(display_image.shape, (4, 4, 3))
            self.assertTrue((display_image == 0).all())

            # This should translate to (2, 2) in the original scale
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 1})
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(2, 2)])
            self.assertEqual(selector_box.crosshair_type, [1])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[1, 1, 1] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Click on a new point
            selector_box._handle_click({'dataX': 2, 'dataY': 2, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1, 0])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[1, 1, 1] = should_be[2, 2, 1] = 255
            display_image = np.array(selector_box.display)
            self.assertTrue((display_image == should_be).all())

            # Unclick
            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 0})
            self.assertEqual(selector_box.idxs, [1])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [0])
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            should_be[2, 2, 1] = 255
            display_image = np.array(selector_box.display)
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
            display_image = np.array(selector_box.display)
            should_be = np.zeros((4, 4, 3), dtype=np.uint8)
            self.assertTrue((display_image == should_be).all())

            # Check it doesn't throw an error if there's no samples selected
            sample_values, sample_types = selector_box.get_samples()
            self.assertEqual(sample_values.shape, (0, 3))
            self.assertEqual(sample_types.shape, (0,))

    def test_selector_box_make_crosshairs(self):
        with self.with_images() as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, output_size=(4, 4), marker_choice='box', w=4,
            )
            crosshair, shift = selector_box.make_crosshairs(4, 4)

            self.assertEqual(crosshair.shape, (4, 4, 4))

            should_be = np.array([
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255]
            ], dtype=np.uint8)
            zeros = np.zeros_like(should_be)
            should_be = np.stack(
                [zeros, should_be, zeros, should_be // 255], axis=2
            )
            self.assertTrue((crosshair == should_be).all())

            self.assertEqual(shift.shape, (4, 4, 4))
            should_be = np.array([
                [255,   0,   0, 255],
                [  0, 255, 255,   0],
                [  0, 255, 255,   0],
                [255,   0,   0, 255],
            ], dtype=np.uint8)
            should_be = np.stack(
                [zeros, should_be, zeros, should_be // 255], axis=2
            )
            self.assertTrue((shift == should_be).all(), shift)

            selector_box = v2v_nb.SelectorBox(
                loader, output_size=(4, 4), marker_choice='cross', w=4,
            )
            crosshair, shift = selector_box.make_crosshairs(3, 3)

            self.assertEqual(crosshair.shape, (3, 3, 4))
            should_be = np.array([
                [  0, 255,   0],
                [255, 255, 255],
                [  0, 255,   0],
            ], dtype=np.uint8)
            zeros = np.zeros_like(should_be)
            should_be = np.stack(
                [zeros, should_be, zeros, should_be // 255], axis=2
            )
            self.assertTrue((crosshair == should_be).all())
            self.assertTrue(crosshair is shift)

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
                    cache_path=crosshair_path,
                )
                display_image = np.array(selector_box.display)

        self.assertEqual(display_image.shape, (8, 8, 3))
        should_be = np.array([
            [  0, 255,   0,   0,   0,   0,   0,   0],
            [255, 255,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0, 255, 255],
            [  0,   0,   0,   0,   0,   0, 255,   0],
        ])
        should_be = np.stack(
            [np.zeros_like(should_be), should_be, np.zeros_like(should_be)],
            axis=2
        )
        should_be[0, 0, :] = 255
        self.assertTrue((display_image == should_be).all())

    def test_selector_box_with_autolocator(self):
        # These were acquired manually.
        sample_pts_1 = np.array([
            [2837, 1809], [2452, 1768], [2114, 1755], [1804, 1721],
            [2776, 2112], [2412, 2052], [2054, 2038], [1736, 1991]
        ])
        sample_pts_2 = np.array([
            [2553, 1647], [2337, 1653], [2114, 1660], [1912, 1653],
            [2513, 1836], [2310, 1849], [2094, 1856], [1858, 1863]
        ])
        marker_pts = np.array([
            [[2681, 1987], [2650, 1992], [2652, 1953], [2683, 1947]],
            [[1727, 2123], [1682, 2129], [1679, 2081], [1724, 2075]],
            [[1692, 1402], [1647, 1403], [1645, 1354], [1691, 1354]],
            [[2679, 1383], [2650, 1384], [2649, 1344], [2678, 1342]]
        ])
        line_op = v2v.AutoLinearize(
            [0, 1, 2, 3],
            marker_pts,
            sample_pts_2,
            np.zeros((4, 3), dtype=np.float32),
        )

        image_path = os.path.join(
            os.path.dirname(__file__), 'data/marker_sample_1.jpg'
        )
        loader = v2v.Loader(image_path, expected_size=(4608, 3456))

        selector_box = v2v_nb.SelectorBox(loader, auto_op=line_op)

        self.assertEqual(selector_box.idxs, list(range(8)))
        self.assertEqual(selector_box.crosshair_type, [0 for _ in range(8)])
        dist = np.linalg.norm(sample_pts_1 - selector_box.crosshairs, axis=1)
        self.assertTrue(dist.max() < 50)

    def test_selector_box_with_border_margin(self):
        with self.with_images() as loader:
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=3, output_size=(4, 4), t=1,
            )

            selector_box._handle_click({'dataX': 1, 'dataY': 1, 'shiftKey': 1})
            self.assertEqual(selector_box.idxs, [])
            self.assertEqual(selector_box.crosshairs, [])
            self.assertEqual(selector_box.crosshair_type, [])

            selector_box._handle_click({'dataX': 2, 'dataY': 2, 'shiftKey': 1})
            self.assertEqual(selector_box.idxs, [0])
            self.assertEqual(selector_box.crosshairs, [(4, 4)])
            self.assertEqual(selector_box.crosshair_type, [1])

    def test_selector_box_with_caching(self):
        with tempfile.TemporaryDirectory() as temp_root:
            cache_path = os.path.join(temp_root, 'cache.json')
            with self.with_images() as loader:
                selector_box = v2v_nb.SelectorBox(
                    loader, w=1, cache_path=cache_path, border_margin=0,
                )
                self.assertEqual(selector_box.idxs, [])
                self.assertEqual(selector_box.crosshairs, [])
                self.assertEqual(selector_box.crosshair_type, [])

                selector_box._handle_click(
                    {'dataX': 1, 'dataY': 1, 'shiftKey': 1}
                )
                self.assertEqual(selector_box.idxs, [0])
                self.assertEqual(selector_box.crosshairs, [(1, 1)])
                self.assertEqual(selector_box.crosshair_type, [1])

                selector_box = v2v_nb.SelectorBox(
                    loader, w=1, cache_path=cache_path,
                )
                self.assertEqual(selector_box.idxs, [0])
                self.assertEqual(selector_box.crosshairs, [(1, 1)])
                self.assertEqual(selector_box.crosshair_type, [1])

    def test_selector_box_with_copy_from(self):
        with self.with_images(True) as (loader_0, loader_1):
            selector_box = v2v_nb.SelectorBox(loader_0, w=1)
            selector_box.idxs = [0, 1]
            selector_box.crosshairs = [(2, 2), (6, 6)]
            selector_box.crosshair_type = [0, 1]

            selector_box = v2v_nb.SelectorBox(
                loader_1, copy_from=selector_box, w=1,
            )

            self.assertEqual(selector_box.idxs, [0, 1])
            self.assertEqual(selector_box.crosshairs, [(2, 2), (6, 6)])
            self.assertEqual(selector_box.crosshair_type, [0, 1])


class ProcessingTest(unittest.TestCase):
    @contextmanager
    def assert_prints(self, message: str, extra: Optional[str] = None):
        buff = io.StringIO()
        with redirect_stdout(buff):
            yield
        disp = f"{buff.getvalue()} {'' if extra is None else extra}"
        self.assertTrue(buff.getvalue().startswith(message), disp)

    @contextmanager
    def with_image(self, as_loader: bool = True, w: int = 8):
        r = w // 2
        with tempfile.TemporaryDirectory() as temp_root:
            image = np.empty((w, w, 3), dtype=np.uint8)
            image[:r, :r, :] = 0
            image[:r, r:, :] = 64
            image[r:, :r, :] = 128
            image[r:, r:, :] = 255

            path = os.path.join(temp_root, 'image.png')
            Image.fromarray(image).save(path)

            csv_path = os.path.join(temp_root, 'values.csv')
            _create_dummy_csv(csv_path, [0, 2**(-0.75), 2**(-0.5), 1])

            if as_loader:
                yield v2v.Loader(path, expected_size=(w, w)), csv_path
            else:
                yield path, csv_path

    def test_build_and_run_alignment_pipeline_from_scratch(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config = v2v_nb.get_config()
            v2v_nb.clear_all()

            root = os.path.abspath(os.path.dirname(__file__))
            config['align_pipe_path'] = os.path.join(
                root, '../data/still_alignment_pipeline.json'
            )
            config['uv_path'] = os.path.join(root, 'data/uv_sample_2.jpg')
            config['vis_path'] = os.path.join(root, 'data/vis_sample_2.jpg')
            config['uv_aligned_path'] = temp_root

            self.assertFalse(os.path.exists(config.out_path))

            for k in [
                'align_pipe_path', 'uv_aligned_path', 'uv_path', 'vis_path'
            ]:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.build_and_run_alignment_pipeline()
                config[k] = v

            with self.assert_prints('Pipeline complete'):
                v2v_nb.build_and_run_alignment_pipeline()

            self.assertTrue(os.path.exists(config.out_path))
            time_made = os.path.getmtime(config.out_path)
            self.assertTrue(config['coe'] is not None)
            self.assertEqual(config['shift'], 0)

            with self.assert_prints('Pipeline already ran'):
                v2v_nb.build_and_run_alignment_pipeline()

            self.assertEqual(time_made, os.path.getmtime(config.out_path))

    def test_build_and_run_alignment_pipeline_with_cached_warp(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config = v2v_nb.get_config()
            v2v_nb.clear_all()

            root = os.path.abspath(os.path.dirname(__file__))
            config['align_pipe_path'] = os.path.join(
                root, '../data/still_alignment_pipeline.json'
            )
            config['uv_path'] = os.path.join(root, 'data/uv_sample.jpg')
            config['vis_path'] = os.path.join(root, 'data/vis_sample.jpg')
            config['uv_aligned_path'] = temp_root

            config['coe'] = np.eye(3)
            with self.assert_prints('Found spatial warp but not'):
                v2v_nb.build_and_run_alignment_pipeline()
            config['shift'] = 0

            with self.assert_prints('Pipeline complete'):
                v2v_nb.build_and_run_alignment_pipeline()

            self.assertTrue(os.path.exists(config.out_path))
            self.assertTrue((config['coe'] == np.eye(3)).all())
            self.assertEqual(config['shift'], 0)

    def test_build_and_run_alignment_pipeline_with_directory_paths(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config = v2v_nb.get_config()
            v2v_nb.clear_all()

            root = os.path.abspath(os.path.dirname(__file__))
            config['align_pipe_path'] = os.path.join(
                root, '../data/still_alignment_pipeline.json'
            )
            uv_root = os.path.join(temp_root, 'UV')
            os.makedirs(uv_root)
            config['uv_path'] = uv_root
            shutil.copyfile(
                os.path.join(root, 'data/uv_sample_2.jpg'),
                os.path.join(uv_root, 'x.jpg')
            )
            vis_root = os.path.join(temp_root, 'VIS')
            os.makedirs(vis_root)
            config['vis_path'] = vis_root
            shutil.copyfile(
                os.path.join(root, 'data/vis_sample_2.jpg'),
                os.path.join(vis_root, 'x.jpg')
            )
            config['uv_aligned_path'] = os.path.join(temp_root, 'UV_align')

            self.assertFalse(os.path.exists(config.out_path))

            with self.assert_prints('Pipeline complete'):
                v2v_nb.build_and_run_alignment_pipeline()

            self.assertTrue(os.path.exists(config.out_path))
            self.assertTrue(config['coe'] is not None)
            self.assertEqual(config['shift'], 0)

            with self.assert_prints('Pipeline already ran'):
                v2v_nb.build_and_run_alignment_pipeline()

    def test_build_and_run_full_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_root:
            config = v2v_nb.get_config()
            v2v_nb.clear_all()

            with self.assert_prints('Alignment must be run'):
                v2v_nb.build_and_run_full_pipeline(None)

            config['coe'], config['shift'] = np.eye(3), 0

            with self.assert_prints('Linearization operator must be built'):
                v2v_nb.build_and_run_full_pipeline(None)

            line_op = v2v.PowerLaw([
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
            ])

            root = os.path.abspath(os.path.dirname(__file__))
            config['align_pipe_path'] = os.path.join(
                root, '../data/still_alignment_pipeline.json'
            )
            config['uv_path'] = os.path.join(root, 'data/uv_sample.jpg')
            config['vis_path'] = os.path.join(root, 'data/vis_sample.jpg')
            config['uv_aligned_path'] = os.path.join(temp_root, 'uv_aligned')
            config['animal_out_path'] = os.path.join(temp_root, 'animal')
            config['human_out_path'] = os.path.join(temp_root, 'human')
            config['animal_sensitivity_path'] = os.path.join(
                root, '../data/animal_sensitivities/apis_sensitivities.csv'
            )
            config['sense_converter_path'] = os.path.join(
                root, '../data/converters/apis_converter.json'
            )

            for k in [
                'align_pipe_path', 'uv_path', 'vis_path', 'animal_out_path',
                'human_out_path', 'sense_converter_path',
                'animal_sensitivity_path',
            ]:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.build_and_run_full_pipeline(line_op)
                config[k] = v

            with self.assert_prints('Pipeline complete'):
                v2v_nb.build_and_run_full_pipeline(line_op)

            self.assertEqual(len(os.listdir(config['animal_out_path'])), 1)
            self.assertEqual(len(os.listdir(config['human_out_path'])), 1)

    def test_build_and_save_alignment_pipeline(self):
        v2v_nb.clear_all()
        config = v2v_nb.get_config()

        with self.assert_prints('Warp operator must be built'):
            v2v_nb.build_and_save_alignment_pipeline(None)

        coe = np.array([[0, -1, 4], [1, 0, 0], [0, 0, 1]])
        warp_op = v2v.Warp(coe=coe, output_size=(8, 8))

        with self.assert_prints('Please specify path to'):
            v2v_nb.build_and_save_alignment_pipeline(warp_op)

        with tempfile.TemporaryDirectory() as temp_root:
            pipe_path = os.path.join(temp_root, 'pipe.json')
            with open(pipe_path, 'w') as out_file:
                out_file.write(' ')

            config['save_align_pipe_path'] = pipe_path
            with self.assert_prints('The alignment pipeline already exists'):
                v2v_nb.build_and_save_alignment_pipeline(warp_op)

        with tempfile.TemporaryDirectory() as temp_root:
            pipe_path = os.path.join(temp_root, 'pipe.json')
            config['save_align_pipe_path'] = pipe_path
            config['build_video_pipeline'] = True
            with self.assert_prints('Please specify path to visible'):
                v2v_nb.build_and_save_alignment_pipeline(warp_op)
            with self.with_image(as_loader=False) as (path, _):
                config['vis_path'] = path
                with self.assert_prints('Done!'):
                    v2v_nb.build_and_save_alignment_pipeline(warp_op)

    def test_build_and_save_autolinearizer(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.with_image(w=11) as (loader, _):
            corners = np.array([
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                [[10, 0], [11, 0], [11, 1], [10, 1]],
                [[0, 10], [1, 10], [1, 11], [0, 11]],
                [[10, 10], [11, 10], [11, 11], [10, 11]]
            ])
            t = 0
            selector = v2v_nb.SelectorBox(loader, w=1)

            with self.assert_prints('Please run ARUCO marker'):
                v2v_nb.build_and_save_autolinearizer(None, None, None)
            with self.assertRaises(RuntimeError):
                v2v_nb.build_and_save_autolinearizer(corners, None, None)
            with self.assertRaises(RuntimeError):
                v2v_nb.build_and_save_autolinearizer(corners[:1], t, None)
            with self.assert_prints('Please select sample locations'):
                v2v_nb.build_and_save_autolinearizer(corners, t, None)
            with self.assert_prints('Please specify'):
                v2v_nb.build_and_save_autolinearizer(corners, t, selector)

            with tempfile.TemporaryDirectory() as temp_root:
                config['camera_path'] = os.path.join(temp_root, 'camera.csv')
                _create_dummy_csv(config['camera_path'], 1. / 401)

                config['linearization_values_path'] = os.path.join(
                    temp_root, 'values.csv'
                )
                _create_dummy_csv(config['linearization_values_path'], 1)

                config['save_auto_op_path'] = config['camera_path']
                with self.assert_prints('Autolinearizer already exists'):
                    v2v_nb.build_and_save_autolinearizer(corners, t, selector)

                auto_op_path = os.path.join(temp_root, 'autolinearizer.json')
                config['save_auto_op_path'] = auto_op_path
                selector.crosshairs = [(2, 2), (8, 2), (8, 8)]
                with self.assert_prints('Selected 3 but have values for 4'):
                    v2v_nb.build_and_save_autolinearizer(corners, t, selector)

                selector.crosshairs = [(2, 2), (8, 2), (8, 8), (2, 8)]
                with self.assert_prints('Done!'):
                    v2v_nb.build_and_save_autolinearizer(corners, t, selector)

                self.assertTrue(os.path.exists(auto_op_path))
                autoline_op = v2v_nb.utils.load_operator(auto_op_path)

        self.assertEqual(autoline_op.marker_ids.tolist(), [0, 1, 2, 3])
        self.assertTrue(np.isclose(autoline_op.marker_points, corners).all())
        sample_points = np.array(selector.crosshairs)
        self.assertTrue(
            np.isclose(autoline_op.sample_points, sample_points).all()
        )
        self.assertTrue(np.isclose(autoline_op.expected_values, 1).all())

    def test_build_and_save_sense_converter(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with tempfile.TemporaryDirectory() as temp_root:
            config['camera_path'] = os.path.join(temp_root, 'camera.csv')
            config['animal_sensitivity_path'] = os.path.join(
                temp_root, 'animals.csv'
            )
            config['reflectivity_path'] = os.path.join(temp_root, 'ref.csv')
            config['save_converter_path'] = os.path.join(
                temp_root, 'conv.json'
            )
            for k in [
                'camera_path', 'animal_sensitivity_path', 'reflectivity_path',
                'save_converter_path'
            ]:
                temp, config[k] = config[k], None
                with self.assert_prints('Please specify'):
                    v2v_nb.build_and_save_sense_converter()
                config[k] = temp

            with open(config['save_converter_path'], 'w') as out_file:
                out_file.write(' ')
            with self.assert_prints('Converter already exists'):
                v2v_nb.build_and_save_sense_converter()
            os.remove(config['save_converter_path'])

            _create_dummy_csv(config['camera_path'], 1)
            _create_dummy_csv(config['animal_sensitivity_path'], 1)
            ref = np.random.uniform(0, 1, (401, 20))
            _create_dummy_csv(config['reflectivity_path'], ref)

            temp_path = os.path.join(temp_root, 'temp.csv')
            for k in [
                'camera_path', 'animal_sensitivity_path', 'reflectivity_path',
            ]:
                temp, config[k] = config[k], temp_path
                with self.assert_prints('Could not find'):
                    v2v_nb.build_and_save_sense_converter()
                config[k] = temp

            with self.assert_prints('Done!'):
                cam_s, an_s, tab = v2v_nb.build_and_save_sense_converter()

        self.assertTrue(isinstance(cam_s, np.ndarray))
        self.assertEqual(cam_s.shape, (401, 4))
        self.assertTrue((cam_s == 1).all())

        self.assertTrue(isinstance(an_s, np.ndarray))
        self.assertEqual(an_s.shape, (401, 4))
        self.assertTrue((an_s == 1).all())

    def test_build_coarse_warp(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.with_image(as_loader=False, w=32) as (path, _):
            loader = v2v.Loader(path, expected_size=(32, 32))
            uv_selector = v2v_nb.SelectorBox(loader, output_size=(16, 16))
            vis_selector = v2v_nb.SelectorBox(loader, output_size=(16, 16))

            with self.assert_prints('Please run both selectors'):
                v2v_nb.build_coarse_warp(None, None)
            with self.assert_prints('Please run both selectors'):
                v2v_nb.build_coarse_warp(uv_selector, None)
            with self.assert_prints('Please specify path to visible'):
                v2v_nb.build_coarse_warp(vis_selector, uv_selector)
            config['vis_path'] = path
            uv_selector.crosshairs = [(0, 0)]
            with self.assert_prints('You need to select the same number'):
                v2v_nb.build_coarse_warp(vis_selector, uv_selector)
            vis_selector.crosshairs = [(0, 0)]
            with self.assert_prints('At least four tie points'):
                v2v_nb.build_coarse_warp(vis_selector, uv_selector)

            # This should be a 90 degree clockwise rotation
            vis_selector.crosshairs = [(0, 0), (32, 0), (32, 32), (0, 32)]
            uv_selector.crosshairs = [(0, 32), (0, 0), (32, 0), (32, 32)]
            warp_op, display_image = v2v_nb.build_coarse_warp(
                vis_selector, uv_selector
            )

            self.assertTrue(isinstance(warp_op, v2v.Warp))
            rot_90 = np.array([
                [0, -1, 32],
                [1, 0, 0],
                [0, 0, 1]
            ])
            self.assertTrue(np.isclose(warp_op.coe, rot_90).all())
            self.assertEqual(warp_op.output_size, (32, 32))

            display_image = np.array(display_image)
            self.assertEqual(display_image.dtype, np.uint8)
            self.assertEqual(display_image.shape, (16, 32, 3))
            self.assertEqual(display_image[0, 0, 0], 0)
            self.assertEqual(display_image[0, 15, 0], 64)
            self.assertEqual(display_image[15, 0, 0], 128)
            self.assertEqual(display_image[15, 15, 0], 255)
            self.assertEqual(display_image[1, 17, 0], 128)
            self.assertEqual(display_image[1, 30, 0], 0)
            self.assertEqual(display_image[14, 17, 0], 255)
            self.assertEqual(display_image[14, 30, 0], 64)

    def test_build_linearizer(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        root = os.path.abspath(os.path.dirname(__file__))
        config['camera_path'] = os.path.join(
            root, '../data/camera_sensitivities.csv'
        )
        config['is_sony_camera'] = True

        with self.assert_prints('Please select samples'):
            v2v_nb.build_linearizer(None, None)

        with self.with_image() as (loader, values_path):
            config['linearization_values_path'] = values_path

            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(8, 8),
            )
            selector_box.idxs = [0, 1, 2, 3]
            selector_box.crosshairs = [(1, 1), (1, 7), (7, 1), (7, 7)]
            selector_box.crosshair_type = [0, 0, 0, 0]

            for k in [
                'camera_path', 'is_sony_camera', 'linearization_values_path',
            ]:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.build_linearizer(selector_box, selector_box)
                config[k] = v

            line_op = v2v_nb.build_linearizer(selector_box, selector_box)
            self.assertTrue(isinstance(line_op, v2v.PowerLaw))

    def test_create_record(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with (
            tempfile.TemporaryDirectory() as temp_root,
            self.with_image() as (loader, _)
        ):
            with self.assert_prints('Skipping record creation since rec'):
                v2v_nb.create_record(
                    None, None, None, None, None,
                )
            csv_path = os.path.join(temp_root, 'record.csv')
            config['sample_record_path'] = csv_path
            with self.assert_prints('Skipping record creation due to la'):
                v2v_nb.create_record(
                    None, None, None, None, None,
                )
            selector = v2v_nb.SelectorBox(loader, w=3)
            selector.idxs = [0, 1]
            selector.crosshairs = [(1, 1), (6, 6)]
            selector.crosshair_type = [0, 0]
            with self.assert_prints('Skipping record creation due to la'):
                v2v_nb.create_record(
                    None, selector, selector, None, None,
                )
            line_op = v2v.Polynomial([
                [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0],
            ])
            with self.assert_prints('Skipping record creation because p'):
                v2v_nb.create_record(
                    line_op, selector, selector, None, None,
                )
            config['linearization_values_path'] = os.path.join(
                temp_root, 'values.csv'
            )
            with self.assert_prints('Could not find'):
                v2v_nb.create_record(
                    line_op, selector, selector, None, None,
                )
            values = np.linspace(0, 1, 401)
            wl = np.arange(300, 701, 1)
            out = np.stack((wl, values, 0.5 + 0.5 * values), axis=1)
            np.savetxt(
                config['linearization_values_path'], out,
                header='wl,1,2', delimiter=',',
            )
            with self.assert_prints('Skipping record creation because c'):
                v2v_nb.create_record(
                    line_op, selector, selector, None, None,
                )
            config['camera_path'] = os.path.join(
                temp_root, 'camera.csv'
            )
            with self.assert_prints('Could not find'):
                v2v_nb.create_record(
                    line_op, selector, selector, None, None,
                )
            camera = np.full((401,), 1. / 401)
            out = np.stack([wl] + [camera] * 4, axis=1)
            np.savetxt(
                config['camera_path'], out,
                header='wl,U,B,G,R', delimiter=',',
            )
            with self.assert_prints('Record created.'):
                v2v_nb.create_record(
                    line_op, selector, selector, None, None,
                )
            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, 'r') as record_file:
                samples = list(csv.DictReader(record_file))
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(samples[0]), 13)
            self.assertEqual(samples[0]['Training Sample?'], 'True')

            config['test_values_path'] = config['linearization_values_path']
            with self.assert_prints('Record created.'):
                v2v_nb.create_record(
                    line_op, selector, selector, selector, selector,
                )

            with open(csv_path, 'r') as record_file:
                samples = list(csv.DictReader(record_file))
            self.assertEqual(len(samples), 4)
            self.assertEqual(len(samples[0]), 13)
            self.assertEqual(samples[0]['Training Sample?'], 'True')
            self.assertEqual(samples[2]['Training Sample?'], 'False')

            config['animal_sensitivity_path'] = os.path.join(
                temp_root, 'animal.csv'
            )
            senses = np.full((401,), 1. / 401)
            out = np.stack([wl] + [senses] * 2, axis=1)
            np.savetxt(
                config['animal_sensitivity_path'], out,
                header='wl,U,B', delimiter=',',
            )
            sense_converter = v2v.LinearMap(np.ones((4, 2)))
            config['sense_converter_path'] = os.path.join(
                temp_root, 'converter.json'
            )
            with open(config['sense_converter_path'], 'w') as out_file:
                json.dump(sense_converter._to_json(), out_file)

            with self.assert_prints('Record created.'):
                v2v_nb.create_record(
                    line_op, selector, selector, selector, selector,
                )

            with open(csv_path, 'r') as record_file:
                samples = list(csv.DictReader(record_file))
            self.assertEqual(len(samples), 4)
            self.assertEqual(len(samples[0]), 17)
            self.assertEqual(samples[0]['Training Sample?'], 'True')
            self.assertEqual(samples[2]['Training Sample?'], 'False')

    def test_evaluate_conversion(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        root = os.path.abspath(os.path.dirname(__file__))
        config['align_pipe_path'] = os.path.join(
            root, '../data/still_alignment_pipeline.json'
        )
        config['sense_converter_path'] = os.path.join(
            root, '../data/converters/apis_converter.json'
        )
        config['animal_sensitivity_path'] = os.path.join(
            root, '../data/animal_sensitivities/apis_sensitivities.csv'
        )
        config['camera_path'] = os.path.join(
            root, '../data/camera_sensitivities.csv'
        )

        with self.with_image() as (loader, values_path):
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(8, 8),
            )
            selector_box.idxs = [0, 1, 2, 3]
            selector_box.crosshairs = [(1, 1), (1, 7), (7, 1), (7, 7)]
            selector_box.crosshair_type = [0, 0, 0, 0]
            line_op = v2v.PowerLaw([
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
            ])

            with self.assert_prints('Linearizer must be built'):
                v2v_nb.evaluate_conversion(
                    None, values_path, selector_box, selector_box
                )
            with self.assert_prints('Please specify path'):
                v2v_nb.evaluate_conversion(
                    line_op, None, selector_box, selector_box
                )
            with self.assert_prints('Please select samples'):
                v2v_nb.evaluate_conversion(
                    line_op, values_path, selector_box, None
                )
            for k in [
                'sense_converter_path', 'animal_sensitivity_path',
                'camera_path'
            ]:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.evaluate_conversion(
                        line_op, values_path, selector_box, selector_box,
                    )
                config[k] = v

            v2v_nb.evaluate_conversion(
                line_op, values_path, selector_box, selector_box,
            )

    def test_evaluate_samples(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.with_image() as (loader, values_path):
            selector_box = v2v_nb.SelectorBox(
                loader, w=1, border_margin=0, output_size=(8, 8),
            )
            selector_box.idxs = [0, 1, 2, 3]
            selector_box.crosshairs = [(1, 1), (1, 7), (7, 1), (7, 7)]
            selector_box.crosshair_type = [0, 0, 0, 0]
            line_op = v2v.PowerLaw([
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01,
                 0.16736099187966763],
            ])

            with self.assert_prints('Linearizer must be built'):
                v2v_nb.evaluate_samples(
                    None, values_path, selector_box, selector_box
                )
            with self.assert_prints('Please specify path'):
                v2v_nb.evaluate_samples(
                    line_op, None, selector_box, selector_box
                )
            with self.assert_prints('Please specify'):
                v2v_nb.evaluate_samples(
                    line_op, values_path, selector_box, selector_box,
                )
            root = os.path.abspath(os.path.dirname(__file__))
            config['camera_path'] = os.path.join(
                root, '../data/camera_sensitivities.csv'
            )
            with self.assert_prints('Please select samples'):
                v2v_nb.evaluate_samples(
                    line_op, values_path, selector_box, None
                )
            v2v_nb.evaluate_samples(
                line_op, values_path, selector_box, selector_box,
            )

    def test_find_and_draw_aruco_markers(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.assert_prints('Please specify '):
            v2v_nb.find_and_draw_aruco_markers()

        root_path = os.path.abspath(os.path.dirname(__file__))
        config['vis_path'] = os.path.join(root_path, 'data/vis_sample.jpg')
        with self.assert_prints('Failed to locate markers.'):
            v2v_nb.find_and_draw_aruco_markers()

        v2v_nb.clear_all()

        config['vis_path'] = os.path.join(
            root_path, 'data/marker_sample_1.jpg'
        )

        corners, t, image = v2v_nb.find_and_draw_aruco_markers()

        marker_pts = np.array([
            [[3141., 2440.], [3069., 2434.], [3072., 2364.], [3144., 2371.]],
            [[1514., 2307.], [1455., 2303.], [1459., 2242.], [1519., 2245.]],
            [[1538., 1384.], [1483., 1384.], [1482., 1322.], [1535., 1323.]],
            [[3153., 1384.], [3080., 1384.], [3081., 1310.], [3154., 1310.]],
        ])
        self.assertEqual(t, 0)
        dist = np.sqrt(((corners - marker_pts)**2).sum(2))
        self.assertTrue(dist.max() < 50)

    def test_make_example_linearization_images(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        root = os.path.abspath(os.path.dirname(__file__))
        config['align_pipe_path'] = os.path.join(
            root, '../data/still_alignment_pipeline.json'
        )
        config['uv_path'] = os.path.join(root, 'data/uv_sample.jpg')
        config['vis_path'] = os.path.join(root, 'data/vis_sample.jpg')

        for k in ['align_pipe_path', 'uv_path', 'vis_path']:
            v, config[k] = config[k], None
            with self.assert_prints('Please specify', k):
                v2v_nb.make_example_linearization_images(None)
            config[k] = v

        with self.assert_prints('Alignment must be run'):
            v2v_nb.make_example_linearization_images(None)

        config['coe'] = np.eye(3)
        with self.assert_prints('Alignment must be run'):
            v2v_nb.make_example_linearization_images(None)
        config['shift'], config['coe'] = 0, None
        with self.assert_prints('Alignment must be run'):
            v2v_nb.make_example_linearization_images(None)
        config['coe'] = np.eye(3)

        with self.assert_prints('Linearization operator must'):
            v2v_nb.make_example_linearization_images(None)

        line_op = v2v.PowerLaw([
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
        ])

        image = v2v_nb.make_example_linearization_images(line_op)
        self.assertTrue(isinstance(image, Image.Image))

    def test_make_final_displaybox_3band(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.with_image(as_loader=False) as (path, _):
            config['animal_out_path'] = config['human_out_path'] = path
            root = os.path.abspath(os.path.dirname(__file__))
            config['animal_sensitivity_path'] = os.path.join(
                root, '../data/animal_sensitivities/apis_sensitivities.csv'
            )

            for k in ['human_out_path', 'animal_sensitivity_path']:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.make_final_displaybox()
                config[k] = v

            config['animal_out_path'] = 'asd'
            with self.assert_prints('Could not find'):
                v2v_nb.make_final_displaybox()
            config['animal_out_path'] = path

            out = v2v_nb.make_final_displaybox()
            self.assertTrue(isinstance(out, v2v_nb.DisplayBox))

    def test_make_final_displaybox_4band(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with tempfile.TemporaryDirectory() as temp_root:
            image = np.empty((8, 8, 3), dtype=np.uint8)
            image[:4, :4, :] = 0
            image[:4, 4:, :] = 64
            image[4:, :4, :] = 128
            image[4:, 4:, :] = 255

            human_path = os.path.join(temp_root, 'human.png')
            Image.fromarray(image).save(human_path)
            animal_path = os.path.join(temp_root, 'animals')
            os.makedirs(animal_path)
            for band in range(4):
                path = os.path.join(animal_path, f'animal_{band}.png')
                Image.fromarray(image[:, :, 0]).save(path)

            config['animal_out_path'] = animal_path
            config['human_out_path'] = human_path
            root = os.path.abspath(os.path.dirname(__file__))
            config['animal_sensitivity_path'] = os.path.join(
                root, '../data/animal_sensitivities/avian_sensitivities.csv'
            )

            for k in ['human_out_path', 'animal_sensitivity_path']:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.make_final_displaybox()
                config[k] = v

            config['animal_out_path'] = 'asd'
            with self.assert_prints('Could not find'):
                v2v_nb.make_final_displaybox()
            config['animal_out_path'] = animal_path

            out = v2v_nb.make_final_displaybox()
            self.assertTrue(isinstance(out, v2v_nb.DisplayBox))

    def test_make_ghostbox(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.assert_prints('Please run alignment'):
            v2v_nb.make_ghostbox()
        config['shift'] = 0

        with self.with_image(as_loader=False) as (path, _):
            config['vis_path'] = config['uv_aligned_path'] = path

            for k in ['vis_path', 'uv_aligned_path']:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.make_ghostbox()
                config[k] = v

            config['vis_path'] = 'asd'
            with self.assert_prints('Could not find file'):
                v2v_nb.make_ghostbox()
            config['vis_path'] = path

            v2v_nb.make_ghostbox()

    def test_make_initial_displaybox(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        with self.assert_prints('Please run alignment'):
            v2v_nb.make_initial_displaybox()
        config['shift'] = 0

        with self.with_image(as_loader=False) as (path, _):
            for k in ['vis_path', 'uv_path', 'uv_aligned_path']:
                config[k] = path

            for k in ['vis_path', 'uv_path', 'uv_aligned_path']:
                v, config[k] = config[k], None
                with self.assert_prints('Please specify', k):
                    v2v_nb.make_initial_displaybox()
                config[k] = v

            config['vis_path'] = 'asd'
            with self.assert_prints('Could not find file'):
                v2v_nb.make_initial_displaybox()
            config['vis_path'] = path

            v2v_nb.make_initial_displaybox()

    def test_make_selectorbox(self):
        config = v2v_nb.get_config()
        v2v_nb.clear_all()

        align_pipe = v2v.Pipeline()
        loader_idx = align_pipe.add_operator(
            v2v.Loader(None, expected_size=(8, 8))
        )
        align_pipe.add_operator(v2v.Loader(None, expected_size=(8, 8)))
        flip_idx = align_pipe.add_operator(v2v.HorizontalFlip())
        writer_idx = align_pipe.add_operator(v2v.Writer(extension='png'))
        align_pipe.add_edge(loader_idx, flip_idx, in_slot=0)
        align_pipe.add_edge(flip_idx, writer_idx, in_slot=0)

        with tempfile.TemporaryDirectory() as temp_root:
            with self.with_image(as_loader=False) as (image_path, _):
                align_path = os.path.join(temp_root, 'align_pipe.json')
                align_pipe.save(align_path)
                config['align_pipe_path'] = align_path
                config['vis_test_path'] = image_path

                with self.assert_prints('Alignment must be run'):
                    v2v_nb.make_selectorbox('vis_test')
                config['shift'], config['coe'] = 0, np.eye(3)

                for k in ['vis_test_path', 'align_pipe_path']:
                    v, config[k] = config[k], None
                    with self.assert_prints('Please specify', k):
                        v2v_nb.make_initial_displaybox()
                    config[k] = v


class UtilsTest(unittest.TestCase):
    def test_coefficient_of_determination(self):
        x = np.random.normal(0, 1, (128,))
        r_2 = v2v_nb.utils.coefficient_of_determination(x, -x)
        self.assertTrue(isclose(r_2, 1.))

        # Test singular case
        x = np.ones(128)
        r_2 = v2v_nb.utils.coefficient_of_determination(x, -x)
        self.assertTrue(isclose(r_2, 1.))


if __name__ == '__main__':
    unittest.main()
