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

        # Check noscale
        x = np.arange(20, dtype=np.uint8).reshape(4, 5)
        x = v2v.utils._coerce_to_image(x, noscale=True)
        self.assertEqual(x.shape, (4, 5, 1))
        self.assertEqual(x.dtype, np.uint8)

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
    def test_detect_motion(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[0:2, :] = 0

        images = np.random.uniform(0, 0.01, (100, 100, 10, 3))
        for t in range(5, 10):
            w, h = (t - 5) * 10, 2 * (t - 4)
            images[:h, :w, t, :] += 0.9

        images = {'image': images, 'mask': mask}

        out = v2v.utils.detect_motion(images)
        should_be = np.array(([False] * 7) + ([True] * 3))

        self.assertTrue((out == should_be).all(), out)

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

    def test_get_photoreceptor_template(self):
        template = v2v.utils.get_photoreceptor_template(500, template='A1')
        self.assertTrue(isinstance(template, np.ndarray))
        self.assertEqual(template.shape, (401,))
        self.assertEqual(template[0], 0.0010322057237818108)
        self.assertEqual(template[100], 0.0016440751261561433)
        self.assertEqual(template[400], 6.854037433359111e-08)

        template = v2v.utils.get_photoreceptor_template(500, template='A2')
        self.assertTrue(isinstance(template, np.ndarray))
        self.assertEqual(template.shape, (401,))
        self.assertEqual(template[0], 0.0005538273213520383)
        self.assertEqual(template[100], 0.001908421897502598)
        self.assertEqual(template[400], 2.1644502203039017e-07)

        with self.assertRaises(ValueError):
            v2v.utils.get_photoreceptor_template(500, template='A3')

    def test_locate_aruco_markers_with_specified_markers(self):
        marker_pts = np.array([
            [[3153., 1384.], [3080., 1384.], [3081., 1310.], [3154., 1310.]],
            [[1538., 1384.], [1483., 1384.], [1482., 1322.], [1535., 1323.]],
            [[1514., 2307.], [1455., 2303.], [1459., 2242.], [1519., 2245.]],
            [[3141., 2440.], [3069., 2434.], [3072., 2364.], [3144., 2371.]],
        ])

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image = v2v.load(os.path.join(data_root, 'marker_sample_1.jpg'))
        # Add an extra channel to check that locate_aruco_markers properly
        # handles images with non-standard numbers of channels.
        image = np.concatenate((image, image), axis=2)
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

    def test_extract_background(self):
        # Check 4-dim.
        image = np.random.uniform(0, 1, (10, 10, 4, 3)).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)
        image = {'image': image, 'mask': mask}
        out = v2v.utils._extract_background(image)
        # Verify it's not the same dictionary
        image['a'] = 1
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertEqual(out['image'].shape, (10, 10, 1, 3))
        should_be = np.mean(image['image'], axis=2, keepdims=True)
        self.assertTrue((np.abs(out['image'] - should_be) < 1e-2).all())

        # Check 3-dim.
        image = np.random.uniform(0, 1, (10, 10, 3)).astype(np.float32)
        image = {'image': image}
        out = v2v.utils._extract_background(image)
        self.assertEqual(out['image'].shape, (10, 10, 3))
        self.assertTrue((np.abs(out['image'] - image['image']) < 1e-2).all())

    def test_prep_for_motion_detection(self):
        for n_c in [1, 2, 3]:
            image = np.zeros((10, 10, n_c), dtype=np.float32)
            image[5, 5, :] = 1.
            mask = np.ones((10, 10), dtype=bool)
            image = {'image': image, 'mask': mask}

            out = v2v.utils._prep_for_motion_detection(image)

            # First, check that this is a different dictionary, and that the
            # image is not the same array.
            image['a'] = 1
            self.assertEqual(out.keys(), {'image', 'mask'})
            # If they're the same array, this will trigger an assert below when
            # we check the value of out['image']
            image['image'][5, 5, :] = 1.

            # Next, check that the image has been converted to grayscale
            self.assertEqual(out['image'].shape, (10, 10, 1))

            # And check the value of the image
            should_be = np.zeros((10, 10, 1), dtype=np.float32)
            should_be[3:8, 3:8, 0] = np.array([
                [ 1.,  4.,  6.,  4.,  1.],
                [ 4., 16., 24., 16.,  4.],
                [ 6., 24., 36., 24.,  6.],
                [ 4., 16., 24., 16.,  4.],
                [ 1.,  4.,  6.,  4.,  1.]
            ])
            should_be /= 256
            self.assertTrue((np.abs(out['image'] - should_be) < 1e-4).all())

    def test_read_jazirrad(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example.JazIrrad')
        wavelengths, response = v2v.utils.read_jazirrad_file(path)
        self.assertTrue(abs(wavelengths[10] - 194.718307) < 1e-3)
        self.assertTrue(abs(response[10] - 0.) < 1e-3)


if __name__ == '__main__':
    unittest.main()
