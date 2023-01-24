import os
import unittest

import numpy as np

import video2vision as v2v


class CoercionTests(unittest.TestCase):
    def test_coerce_to_2dim(self):
        image = np.arange(60).reshape(4, 5, 3)
        mask = np.ones((4, 5), dtype=np.uint8)
        video = np.arange(420).reshape(4, 5, 7, 3)

        x = {'image': image, 'mask': mask}
        with v2v.utils._coerce_to_2dim(x):
            self.assertEqual(x['image'].shape, (20, 3))
            self.assertEqual(x['mask'].shape, (20,))
        self.assertEqual(x['image'].shape, (4, 5, 3))
        self.assertEqual(x['mask'].shape, (4, 5))

        x = {'image': video, 'mask': mask}
        with v2v.utils._coerce_to_2dim(x):
            self.assertEqual(x['image'].shape, (140, 3))
            self.assertEqual(x['mask'].shape, (20,))
        self.assertEqual(x['image'].shape, (4, 5, 7, 3))
        self.assertEqual(x['mask'].shape, (4, 5))

    def test_coerce_to_3dim(self):
        image = np.arange(60).reshape(4, 5, 3)
        mask = np.ones((4, 5), dtype=np.uint8)
        video = np.arange(420).reshape(4, 5, 7, 3)

        x = {'image': image, 'mask': mask}
        with v2v.utils._coerce_to_3dim(x):
            self.assertEqual(x['image'].shape, (4, 5, 3))
            self.assertEqual(x['mask'].shape, (4, 5,))
        self.assertEqual(x['image'].shape, (4, 5, 3))
        self.assertEqual(x['mask'].shape, (4, 5))

        x = {'image': video, 'mask': mask}
        with v2v.utils._coerce_to_3dim(x):
            self.assertEqual(x['image'].shape, (4, 5, 21))
            self.assertEqual(x['mask'].shape, (4, 5,))
        self.assertEqual(x['image'].shape, (4, 5, 7, 3))
        self.assertEqual(x['mask'].shape, (4, 5))

    def test_coerce_to_4dim(self):
        image = np.arange(60).reshape(4, 5, 3)
        video = np.arange(420).reshape(4, 5, 7, 3)

        x = {'image': image}
        with v2v.utils._coerce_to_4dim(x):
            self.assertEqual(x['image'].shape, (4, 5, 1, 3))
        self.assertEqual(x['image'].shape, (4, 5, 3))

        x = {'image': video}
        with v2v.utils._coerce_to_4dim(x):
            self.assertEqual(x['image'].shape, (4, 5, 7, 3))
        self.assertEqual(x['image'].shape, (4, 5, 7, 3))

    def test_coerce_to_dict(self):
        x = v2v.utils._coerce_to_dict(np.arange(4.).reshape(2, 2))
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(x.keys(), {'image'})

        x_2 = v2v.utils._coerce_to_dict(x)
        self.assertTrue(x_2 is x)

    def test_coerce_to_image(self):
        # Check coercion of 2-dimensional images
        x = np.arange(20.).reshape(4, 5)
        x = v2v.utils._coerce_to_image(x)
        self.assertEqual(x.shape, (4, 5, 1))
        self.assertEqual(x.dtype, np.float32)

        # Check coercion of 3-dimensional images
        x = np.arange(20.).reshape(4, 5, 1)
        x = v2v.utils._coerce_to_image(x)
        self.assertEqual(x.shape, (4, 5, 1))
        self.assertEqual(x.dtype, np.float32)

        # Check coercion of 4-dimensional images and uint8
        x = np.arange(240, dtype=np.uint8).reshape(4, 5, 4, 3)
        x = v2v.utils._coerce_to_image(x)
        self.assertEqual(x.shape, (4, 5, 4, 3))
        self.assertEqual(x.dtype, np.float32)

        # Check coercion of 1-dimensional images
        with self.assertRaises(ValueError):
            v2v.utils._coerce_to_image(np.arange(240.))

        # Check coercion of 5-dimensional images
        with self.assertRaises(ValueError):
            v2v.utils._coerce_to_image(np.arange(240.).reshape(4, 5, 1, 4, 3))

        # Check handling of weird dtypes
        x = np.arange(20, dtype=np.int64).reshape(4, 5, 1)
        x = v2v.utils._coerce_to_image(x)
        self.assertEqual(x.dtype, np.float32)

    def test_coerce_to_mask(self):
        # Check coercion of 2-dimensional masks
        x = np.ones((4, 5), dtype=np.uint8)
        x = v2v.utils._coerce_to_mask(x)
        self.assertEqual(x.shape, (4, 5))
        self.assertEqual(x.dtype, np.uint8)

        # Check coercion of bools
        x = np.ones((4, 5), dtype=np.bool_)
        x = v2v.utils._coerce_to_mask(x)
        self.assertEqual(x.shape, (4, 5))
        self.assertEqual(x.dtype, np.uint8)

        # Check coercion of 1-dimensional masks
        with self.assertRaises(ValueError):
            v2v.utils._coerce_to_mask(np.ones(20, dtype=np.uint8))

        # Check coercion of 3-dimensional masks
        with self.assertRaises(ValueError):
            v2v.utils._coerce_to_mask(np.ones((4, 4, 4), dtype=np.uint8))


class UtilitiesTests(unittest.TestCase):
    def test_extract_samples(self):
        # First, image. We choose the size of the image so we can test the
        # out-of-bounds handling.
        image = np.zeros((16, 32, 3), dtype=np.float32)
        image[:8, :8], image[:8, 24:] = (128, 64, 32), (32, 64, 128)
        points = np.array([[4, 4], [28, 4], [8, 4]])
        should_be = np.array([[128, 64, 32], [32, 64, 128], [0, 0, 0]])
        samples = v2v.utils.extract_samples(image, points, 1)
        self.assertTrue((samples == should_be).all())

        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, 28]])
        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, -4]])
        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, 4, 0]])

        # Next, video
        image = np.zeros((16, 32, 2, 3), dtype=np.float32)
        image[:8, :8, 1], image[:8, 24:, 0] = (128, 64, 32), (32, 64, 128)
        points = np.array([[4, 4, 1], [28, 4, 0], [8, 4, 1]])
        should_be = np.array([[128, 64, 32], [32, 64, 128], [0, 0, 0]])
        samples = v2v.utils.extract_samples(image, points, 1)
        self.assertTrue((samples == should_be).all())

        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, 28, 0]])
        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, 4]])
        with self.assertRaises(ValueError):
            v2v.utils.extract_samples(image, [[4, 4, 2]])

    def test_locate_aruco_markers_with_specified_markers(self):
        marker_pts = np.array([
            [[3153., 1384.], [3080., 1384.], [3081., 1310.], [3154., 1310.]],
            [[1538., 1384.], [1483., 1384.], [1482., 1322.], [1535., 1323.]],
            [[1514., 2307.], [1455., 2303.], [1459., 2242.], [1519., 2245.]],
            [[3141., 2440.], [3069., 2434.], [3072., 2364.], [3144., 2371.]],
        ])

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image = v2v.load(os.path.join(data_root, 'marker_sample_1.jpg'))
        image = np.stack((np.zeros_like(image), image), axis=2)

        t, pred_markers = v2v.utils.locate_aruco_markers(
            {'image': image}, [3, 2, 1, 0],
        )
        self.assertEqual(t.shape, (1,))
        self.assertEqual(t, 1)
        self.assertEqual(pred_markers.shape, (1, 4, 4, 2))
        pred_markers = pred_markers.reshape(4, 4, 2)
        dist = np.sqrt(((pred_markers - marker_pts)**2).sum(2))
        self.assertTrue(dist.max() < 50)

    def test_locate_aruco_markers_with_flipped_image(self):
        marker_pts = np.array([
            [[3153., 1384.], [3080., 1384.], [3081., 1310.], [3154., 1310.]],
            [[1538., 1384.], [1483., 1384.], [1482., 1322.], [1535., 1323.]],
            [[1514., 2307.], [1455., 2303.], [1459., 2242.], [1519., 2245.]],
            [[3141., 2440.], [3069., 2434.], [3072., 2364.], [3144., 2371.]],
        ])

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image = v2v.load(os.path.join(data_root, 'marker_sample_1.jpg'))
        image = image[:, ::-1, :]

        marker_pts[:, :, 0] = image.shape[1] - marker_pts[:, :, 0]

        t, pred_markers = v2v.utils.locate_aruco_markers(
            {'image': image}, [3, 2, 1, 0],
        )
        self.assertEqual(t.shape, (1,))
        self.assertEqual(t, 0)
        self.assertEqual(pred_markers.shape, (1, 4, 4, 2))
        pred_markers = pred_markers.reshape(4, 4, 2)
        dist = np.sqrt(((pred_markers - marker_pts)**2).sum(2))
        self.assertTrue(dist.max() < 50)

    def test_locate_aruco_markers_without_specified_markers(self):
        marker_pts = np.array([
            [[3141., 2440.], [3069., 2434.], [3072., 2364.], [3144., 2371.]],
            [[1514., 2307.], [1455., 2303.], [1459., 2242.], [1519., 2245.]],
            [[1538., 1384.], [1483., 1384.], [1482., 1322.], [1535., 1323.]],
            [[3153., 1384.], [3080., 1384.], [3081., 1310.], [3154., 1310.]],
        ])

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image = v2v.load(os.path.join(data_root, 'marker_sample_1.jpg'))
        image = np.stack((np.zeros_like(image), image), axis=2)

        t, (pred_markers,), (marker_ids,), = v2v.utils.locate_aruco_markers(
            {'image': image}
        )
        self.assertEqual(t.shape, (1,))
        self.assertEqual(t, 1)
        self.assertEqual(pred_markers.shape, (4, 4, 2))
        self.assertEqual(marker_ids.shape, (4,))

        marker_pts = np.stack(
            [marker_pts[i] for i in marker_ids.flatten()], axis=0
        )
        pred_markers = pred_markers.reshape(4, 4, 2)
        dist = np.sqrt(((pred_markers - marker_pts)**2).sum(2))
        self.assertTrue(dist.max() < 50)

    def test_read_jazirrad(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example.JazIrrad')
        wavelengths, response = v2v.utils.read_jazirrad_file(path)
        self.assertTrue(abs(wavelengths[10] - 194.718307) < 1e-3)
        self.assertTrue(abs(response[10] - -9.300991) < 1e-3)


if __name__ == '__main__':
    unittest.main()
