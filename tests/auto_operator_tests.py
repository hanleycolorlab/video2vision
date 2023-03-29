import json
import os
import unittest
import warnings

import cv2
import numpy as np

import video2vision as v2v


def _warp_points(coe, xys):
    if coe.shape[0] == 2:
        coe = np.concatenate((coe, np.array([[0, 0, 1]])), axis=0)
    xys = coe.dot(xys.T).T
    return xys / xys[:, 2:3]


class AutoAlignTest(unittest.TestCase):
    def test_handling_with_aruco(self):
        align_op = v2v.AutoAlign(bands=[[0, 1, 2], []], method='aruco')

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image_path = os.path.join(data_root, 'marker_sample_1.jpg')
        image = v2v.load(image_path)

        # Crop and shrink to reduce runtime and max error
        image = image[1000:3500, 1000:3500]
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

        h, w, _ = image.shape
        data_dict = {'image': image}

        # This should be a slight rotation.
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(w, h), scale=1)
        warped_image = cv2.warpAffine(image, coe, (w, h))
        coe = np.concatenate((coe, np.array([[0, 0, 1.]])), axis=0)
        # Include extra key to test it is preserved
        warped_dict = {'image': warped_image, 'dummy': 1}

        # Test on image input
        out_dict = align_op(warped_dict, data_dict)
        self.assertIsNot(align_op.coe, None)
        self.assertEqual(out_dict.get('dummy', 0), 1)

        # Check alignment is correct
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        xs, ys = xs.flatten(), ys.flatten()
        orig_xys = np.stack((xs, ys, np.ones_like(xs)), axis=1)
        warp_xys = _warp_points(coe, orig_xys)
        warp_xys = _warp_points(align_op.coe, warp_xys)
        self.assertTrue((np.abs(warp_xys - orig_xys) < 3).all())
        # When we compare the images, we need to only compare the interiors,
        # since around the edges parts of the image will have been rotated off
        # and then back on again, leaving blank spaces.
        out_crop = out_dict['image'][500:-500, 500:-500]
        image_crop = image[500:-500, 500:-500]
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

        # Now run a second batch through, vertically flipped. If the alignment
        # is reused, this should work.
        warped_dict = {
            'image': cv2.warpAffine(image[::-1], coe[:2, :], (w, h)),
            'dummy': 2,
        }
        out_dict = align_op(warped_dict, data_dict)
        self.assertEqual(out_dict.get('dummy', 0), 2)
        out_crop = out_dict['image'][500:-500, 500:-500].astype(np.int64)
        image_crop = image[::-1][500:-500, 500:-500].astype(np.int64)
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

        # Test that it raises a RuntimeError if pointed to an image with no
        # ARUCO markers in it.
        align_op = v2v.AutoAlign(bands=[[0, 1, 2], []], method='aruco')

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image_path = os.path.join(data_root, 'uv_sample.jpg')
        image = v2v.load(image_path)

        # TODO: Add video test
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(w, h), scale=1)
        warped_image = cv2.warpAffine(image, coe, (w, h))
        with self.assertRaises(RuntimeError):
            align_op(warped_image, data_dict)

    def test_handling_with_ecc(self):
        align_op = v2v.AutoAlign(bands=[[0, 1, 2], []])

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image_path = os.path.join(data_root, 'uv_sample.jpg')
        image = v2v.load(image_path)

        # Resize the image so it runs faster...
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

        h, w, _ = image.shape
        data_dict = {'image': image}

        # This should be a slight rotation.
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(w, h), scale=1)
        warped_image = cv2.warpAffine(image, coe, (w, h))
        coe = np.concatenate((coe, np.array([[0, 0, 1.]])), axis=0)
        mask = np.zeros(warped_image.shape[:2], dtype=np.uint8)
        mask[300:-300, 300:-300] = 1
        # Include extra key to test it is preserved
        warped_dict = {'image': warped_image, 'mask': mask, 'dummy': 1}

        # Test on image input
        out_dict = align_op(warped_dict, data_dict)
        self.assertIsNot(align_op.coe, None)
        self.assertEqual(out_dict.get('dummy', 0), 1)

        # Check alignment is correct
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        xs, ys = xs.flatten(), ys.flatten()
        orig_xys = np.stack((xs, ys, np.ones_like(xs)), axis=1)
        warp_xys = _warp_points(coe, orig_xys)
        warp_xys = _warp_points(align_op.coe, warp_xys)
        self.assertTrue((np.abs(warp_xys - orig_xys) < 0.06).all())
        # When we compare the images, we need to only compare the interiors,
        # since around the edges parts of the image will have been rotated off
        # and then back on again, leaving blank spaces.
        out_crop = out_dict['image'][500:-500, 500:-500]
        image_crop = image[500:-500, 500:-500]
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

        # Check the mask
        inv_coe = cv2.getRotationMatrix2D(angle=-1.2, center=(w, h), scale=1)
        mask_should_be = cv2.warpAffine(mask, inv_coe, (w, h))
        self.assertTrue((out_dict['mask'] != mask_should_be).mean() < 1e-2)

        # Now run a second batch through, vertically flipped. If the alignment
        # is reused, this should work.
        warped_dict = {
            'image': cv2.warpAffine(image[::-1], coe[:2, :], (w, h)),
            'dummy': 2,
        }
        out_dict = align_op(warped_dict, data_dict)
        self.assertEqual(out_dict.get('dummy', 0), 2)
        out_crop = out_dict['image'][500:-500, 500:-500].astype(np.int64)
        image_crop = image[::-1][500:-500, 500:-500].astype(np.int64)
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

        # TODO: Add video test

    def test_serialization(self):
        align_op = v2v.AutoAlign(bands=[[0], [0]], mask=[1, 1, -1, -1])

        op_dict = json.loads(json.dumps(align_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        align_op_2 = op_class(**op_dict)

        self.assertIs(align_op_2.coe, None)
        self.assertIs(align_op_2.output_size, None)

        image_1 = np.arange(10, dtype=np.uint8).repeat(10).reshape(10, 10)
        image_2 = np.arange(1, 11, dtype=np.uint8).repeat(10).reshape(10, 10)

        out_1 = align_op({'image': image_1}, {'image': image_2})
        out_2 = align_op_2({'image': image_1}, {'image': image_2})

        self.assertTrue((out_1['image'] == out_2['image']).all())
        self.assertTrue((align_op.coe == align_op_2.coe).all())
        self.assertEqual(align_op.output_size, align_op_2.output_size)

        op_dict = json.loads(json.dumps(align_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        align_op_3 = op_class(**op_dict)

        self.assertTrue((align_op.coe == align_op_3.coe).all())
        self.assertEqual(align_op.output_size, tuple(align_op_3.output_size))

        out_1 = align_op({'image': image_2}, {'image': image_1})
        out_2 = align_op_3({'image': image_2}, {'image': image_1})
        self.assertTrue((out_1['image'] == out_2['image']).all())


class AutoLinearize(unittest.TestCase):
    def _build_image(self, a: int, b: int, c: int, d: int):
        image = np.full((500, 500, 1), 255, dtype=np.float32)
        # Place ARUCO marks on image
        uls = [[10, 10], [10, 440], [440, 10], [440, 440]]
        for i, (x, y) in enumerate(uls):
            # Name of this function changed in cv2 version 4.7.0
            if v2v.utils._CV2_VERSION < (4, 7, 0):
                mark = cv2.aruco.drawMarker(
                    v2v.utils._MARKER_DICTIONARY, i, 50
                )
            else:
                mark = cv2.aruco.generateImageMarker(
                    v2v.utils._MARKER_DICTIONARY, i, 50
                )
            image[y:y + 50, x:x + 50, :] = mark.reshape(50, 50, 1)
        # Add patches to sample. The polynomial we are trying to fit will be
        # p(x) = x^2 - 1.
        image[150:200, 150:200, :], image[300:350, 150:200, :] = a, b
        image[150:200, 300:350, :], image[300:350, 300:350, :] = c, d

        marker_pts = np.array([
            [ 10., 489.], [ 10., 440.], [ 59., 440.], [ 59., 489.],
            [440., 489.], [440., 440.], [489., 440.], [489., 489.],
            [ 10.,  59.], [ 10.,  10.], [ 59.,  10.], [ 59.,  59.],
            [440.,  59.], [440.,  10.], [489.,  10.], [489.,  59.],
        ])
        # And the original sample points:
        sample_pts = np.array([[325, 175], [175, 175], [175, 325], [325, 325]])

        return image, marker_pts, sample_pts

    def test_handling_polynomial(self):
        '''
        Basic test using polynomials.
        '''
        image, marker_pts, sample_pts = self._build_image(5, 10, 15, 4)
        exp_values = np.array([[15], [224], [24], [99]])
        image = np.stack((np.zeros_like(image), image), axis=2)
        # Include a dummy value to test it's preserved
        image = {'image': image, 'dummy': 1}

        # Build the pipeline.
        line_op = v2v.AutoLinearize(
            marker_ids=[0, 1, 2, 3],
            marker_points=marker_pts,
            sample_points=sample_pts,
            expected_values=exp_values,
            order=2,
            sample_width=5,
            method='poly',
        )

        # Put the data through the operator.
        image = line_op(image)
        self.assertEqual(image.get('dummy', 0), 1)

        coefs = np.array(line_op.op.funcs[0].coef)
        self.assertTrue(np.abs(coefs - np.array([-1, 0, 1])).max() < 1e-5)

        # Run second batch through to verify it reuses the polynomial.
        image = np.ones((500, 500, 1), dtype=np.float32)
        out = line_op({'image': image, 'dummy': 2})
        self.assertEqual(out.get('dummy', 0), 2)
        self.assertTrue((np.abs(out['image']) <= 1e-2).all())

    def test_handling_powerlaw(self):
        '''
        Basic test using a power law.
        '''
        image, marker_pts, sample_pts = self._build_image(1, 2, 3, 4)
        exp_values = np.array([[161], [53], [5], [17]])
        image = np.stack((np.zeros_like(image), image), axis=2)

        # Build the pipeline.
        line_op = v2v.AutoLinearize(
            marker_ids=[0, 1, 2, 3],
            marker_points=marker_pts,
            sample_points=sample_pts,
            expected_values=exp_values,
            order=2,
            sample_width=5,
            method='power',
        )

        # Put the data through the operator. This can raise a harmless
        # RuntimeWarning caused by an overflow.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'overflow encountered')
            image = line_op(image)

        self.assertTrue(abs(line_op.op.funcs[0].scale - 2) < 1e-5,
                        line_op.op.funcs[0])
        self.assertTrue(abs(line_op.op.funcs[0].base - 3) < 1e-5)
        self.assertTrue(abs(line_op.op.funcs[0].shift + 1) < 1e-5)

        # Run second batch through to verify it reuses the polynomial.
        image = np.ones((500, 500, 1), dtype=np.float32)
        out = line_op(image)
        self.assertTrue((np.abs(out['image'] - 5) <= 1e-2).all())

    def test_sampling(self):
        '''
        Tests the :func:`extract_samples` function.
        '''
        # TODO: Add video
        dist_image = np.zeros((20, 20, 1), dtype=np.float32)
        dist_image[0:10, 0:10], dist_image[0:10, 10:20] = 4, 8
        dist_image[10:20, 0:10], dist_image[10:20, 10:20] = 2, 6
        image = dist_image**2 - dist_image
        data_dict = {'image': image}
        dist_dict = {'image': dist_image}

        pts = [(5, 5), (15, 5), (5, 15), (15, 15)]
        samples = v2v.utils.extract_samples(
            data_dict['image'], pts, width=1
        )
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.shape, (4, 1))
        should_be = np.array([12, 56, 2, 30]).reshape(4, 1)
        self.assertTrue((samples == should_be).all(), samples)
        samples = v2v.utils.extract_samples(
            dist_dict['image'], pts, width=1
        )
        should_be = np.array([4, 8, 2, 6]).reshape(4, 1)
        self.assertTrue((samples == should_be).all(), samples)

    def test_locate_samples(self):
        '''
        Tests the :meth:`_locate_samples` method.
        '''
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

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        image = v2v.load(os.path.join(data_root, 'marker_sample_1.jpg'))

        pred_samples, _ = line_op._locate_samples({'image': image})
        dist = np.linalg.norm(sample_pts_1 - pred_samples, axis=1)

        self.assertTrue(dist.max() < 50)

    def test_serializing_polynomial(self):
        '''
        Basic test using polynomials.
        '''
        image, marker_pts, sample_pts = self._build_image(5, 10, 15, 4)
        exp_values = np.array([[15], [224], [24], [99]])

        # Build the pipeline.
        line_op = v2v.AutoLinearize(
            marker_ids=[0, 1, 2, 3],
            marker_points=marker_pts,
            sample_points=sample_pts,
            expected_values=exp_values,
            order=2,
            sample_width=5,
            method='poly',
        )

        op_dict = json.loads(json.dumps(line_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        line_op_2 = op_class(**op_dict)

        self.assertIs(line_op_2.op, None)

        # Put the data through the operator.
        out_1 = line_op(image)
        out_2 = line_op_2(image)

        self.assertTrue((out_1['image'] == out_2['image']).all())
        coefs_1 = np.array(line_op.op.funcs[0].coef)
        coefs_2 = np.array(line_op_2.op.funcs[0].coef)
        self.assertTrue((coefs_1 == coefs_2).all())

        op_dict = json.loads(json.dumps(line_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        line_op_3 = op_class(**op_dict)

        coefs_3 = np.array(line_op_3.op.funcs[0].coef)
        self.assertTrue((coefs_1 == coefs_3).all())

        image = np.random.normal(0, 1, (10, 10, 1))

        out_1 = line_op({'image': image})
        out_2 = line_op_3({'image': image})
        self.assertTrue((out_1['image'] == out_2['image']).all())

    def test_serializing_powerlaw(self):
        '''
        Basic test using polynomials.
        '''
        image, marker_pts, sample_pts = self._build_image(1, 2, 3, 4)
        exp_values = np.array([[161], [53], [5], [17]])

        # Build the pipeline.
        line_op = v2v.AutoLinearize(
            marker_ids=[0, 1, 2, 3],
            marker_points=marker_pts,
            sample_points=sample_pts,
            expected_values=exp_values,
            order=2,
            sample_width=5,
            method='power',
        )

        op_dict = json.loads(json.dumps(line_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        line_op_2 = op_class(**op_dict)

        self.assertIs(line_op_2.op, None)

        # Put the data through the operator.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'overflow encountered')
            out_1 = line_op(image)
            out_2 = line_op_2(image)

        self.assertTrue((out_1['image'] == out_2['image']).all())
        self.assertEqual(repr(line_op.op.funcs), repr(line_op_2.op.funcs))

        op_dict = json.loads(json.dumps(line_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        line_op_3 = op_class(**op_dict)

        self.assertEqual(repr(line_op.op.funcs), repr(line_op_3.op.funcs))

        image = np.random.normal(0, 1, (10, 10, 1))

        out_1 = line_op({'image': image})
        out_2 = line_op_3({'image': image})
        self.assertTrue((out_1['image'] == out_2['image']).all())


class AutoTemporalAlignTest(unittest.TestCase):
    def _build_image(self):
        image = np.zeros((128, 128, 3), dtype=np.float32)
        colors = [(192, 0, 0), (0, 192, 128), (0, 0, 192), (192, 192, 192)]

        for x, c in zip([64, 32, 16, 8, 4], colors):
            image[x:-x, x:-x, :] = c

        return image / 256

    def test_handling(self):
        align_op = v2v.AutoTemporalAlign((0, 2), bands=[[0, 1, 2], []])

        image = self._build_image()
        data_dict = {
            'image': np.stack((np.zeros_like(image), image), axis=2),
            'names': ['a', 'b'],
        }

        # This should be a slight rotation.
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(64, 64), scale=1)
        warped_image = cv2.warpAffine(image, coe, (128, 128))
        coe = np.concatenate((coe, np.array([[0, 0, 1.]])), axis=0)
        warped_image = np.stack((warped_image, warped_image + 0.2), axis=2)
        # Include dummy value to test it's preserved
        warped_dict = {
            'image': warped_image,
            'dummy': 1,
            'names': ['c', 'd'],
        }

        # Test on image input
        out_dict = align_op(warped_dict, data_dict)
        self.assertEqual(out_dict['image'].shape, (128, 128, 1, 3))
        self.assertEqual(out_dict.get('dummy', 0), 1)
        self.assertEqual(out_dict.get('names', 0), ['c-b'])

        # Check time shift is correct
        self.assertEqual(align_op.time_shift, 1)
        self.assertEqual(align_op.buff.shape, (128, 128, 1, 3))
        self.assertEqual(align_op.buff_names, ['d'])
        self.assertTrue(
            (align_op.buff == warped_dict['image'][:, :, 1:2]).all()
        )

        # Check alignment is correct
        xs, ys = np.meshgrid(np.arange(128), np.arange(128))
        xs, ys = xs.flatten(), ys.flatten()
        orig_xys = np.stack((xs, ys, np.ones_like(xs)), axis=1)
        warp_xys = _warp_points(coe, orig_xys)
        warp_xys = _warp_points(align_op.coe, warp_xys)
        self.assertTrue((np.abs(warp_xys - orig_xys) < 1e-2).all())
        # When we compare the images, we need to only compare the interiors,
        # since around the edges parts of the image will have been rotated off
        # and then back on again, leaving blank spaces.
        out_crop = out_dict['image'][32:-32, 32:-32, 0]
        image_crop = image[32:-32, 32:-32]
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

        # Test second batch
        warped_dict = {
            'image': warped_image + 32, 'dummy': 2, 'names': ['e', 'f'],
        }
        out_dict = align_op(warped_dict, data_dict)
        self.assertEqual(out_dict.get('dummy', 0), 2)
        out_crop = out_dict['image'][32:-32, 32:-32, 0].astype(np.int64)
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)
        out_crop = out_dict['image'][32:-32, 32:-32, 1].astype(np.int64)
        self.assertTrue(np.abs(out_crop - 32 - image_crop).mean() <= 1)
        self.assertEqual(out_dict.get('names', 0), ['d-a', 'e-b'])

    def test_without_time_shift(self):
        align_op = v2v.AutoTemporalAlign((0, 2), bands=[[0, 1, 2], []])

        image = self._build_image()
        data_dict = {'image': np.stack((image, -image), axis=2)}

        # This should be a slight rotation.
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(64, 64), scale=1)
        warped_image = cv2.warpAffine(image, coe, (128, 128))
        coe = np.concatenate((coe, np.array([[0, 0, 1.]])), axis=0)
        warped_dict = {
            'image': np.stack((warped_image, -warped_image), axis=2)
        }

        # Test on image input
        out_dict = align_op(warped_dict, data_dict)

        # Check time shift is correct
        self.assertEqual(align_op.time_shift, 0)
        self.assertEqual(out_dict['image'].shape, (128, 128, 2, 3))

        # Check alignment is correct
        xs, ys = np.meshgrid(np.arange(128), np.arange(128))
        xs, ys = xs.flatten(), ys.flatten()
        orig_xys = np.stack((xs, ys, np.ones_like(xs)), axis=1)
        warp_xys = _warp_points(coe, orig_xys)
        warp_xys = _warp_points(align_op.coe, warp_xys)
        self.assertTrue((np.abs(warp_xys - orig_xys) < 1e-2).all())
        # When we compare the images, we need to only compare the interiors,
        # since around the edges parts of the image will have been rotated off
        # and then back on again, leaving blank spaces.
        out_crop = out_dict['image'][32:-32, 32:-32, 0]
        image_crop = image[32:-32, 32:-32]
        self.assertTrue(np.abs(out_crop - image_crop).mean() <= 1)

    def test_motion_detection(self):
        align_op = v2v.AutoTemporalAlign((0, 2), bands=[[0, 1, 2], []])

        self.assertTrue(align_op.source_background is None)
        self.assertTrue(align_op.control_background is None)
        self.assertFalse(align_op.skipped_batch)

        image = self._build_image()
        data_dict = {'image': np.stack([image] * 3, axis=2)}
        data_dict['image'] += np.random.uniform(0, 0.001, (128, 128, 3, 3))

        # This should be a slight rotation.
        coe = cv2.getRotationMatrix2D(angle=1.2, center=(64, 64), scale=1)
        warped_image = cv2.warpAffine(image, coe, (128, 128))
        warped_dict = {'image': np.stack([warped_image] * 3, axis=2)}
        warped_dict['image'] += np.random.uniform(0, 0.001, (128, 128, 3, 3))

        # Test on input
        out = align_op(warped_dict, data_dict)

        # Check that it was held
        self.assertTrue(isinstance(out, v2v.operators.HoldToken))

        # Check background is correct
        self.assertTrue(align_op.skipped_batch)
        self.assertTrue(align_op.source_background is not None)
        self.assertTrue(align_op.control_background is not None)

        # Check ResetPipeline is triggered.
        data_dict = {'image': np.stack((image, -image, -image), axis=2)}
        warped_dict = {
            'image': np.stack((warped_image, -warped_image, -warped_image),
                              axis=2)
        }

        with self.assertRaises(v2v.ResetPipeline):
            out = align_op(warped_dict, data_dict)

        # Check reset method
        align_op.reset()
        self.assertFalse(align_op.skipped_batch)
        self.assertTrue(align_op.source_background is None)
        self.assertTrue(align_op.control_background is None)

    def test_serialization(self):
        align_op = v2v.AutoTemporalAlign(
            (-1, 2), bands=[[0], [0]], mask=[1, 1, -1, -1]
        )

        op_dict = json.loads(json.dumps(align_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        align_op_2 = op_class(**op_dict)

        self.assertIs(align_op_2.coe, None)
        self.assertIs(align_op_2.output_size, None)
        self.assertIs(align_op_2.time_shift, None)

        image_1 = np.arange(0., 1., 0.1, dtype=np.float32)
        image_1 = image_1.repeat(10).reshape(10, 10, 1, 1)
        image_1 = image_1 + np.arange(0., 1., 0.2).reshape(1, 1, 5, 1)
        image_2 = image_1 + 0.2

        out_1 = align_op({'image': image_1}, {'image': image_2})
        out_2 = align_op_2({'image': image_1}, {'image': image_2})

        self.assertTrue((out_1['image'] == out_2['image']).all())
        self.assertTrue((align_op.coe == align_op_2.coe).all())
        self.assertEqual(align_op.output_size, align_op_2.output_size)
        self.assertEqual(align_op.time_shift, align_op_2.time_shift)

        op_dict = json.loads(json.dumps(align_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        align_op_3 = op_class(**op_dict)

        self.assertTrue((align_op.coe == align_op_3.coe).all())
        self.assertEqual(align_op.output_size, tuple(align_op_3.output_size))
        self.assertEqual(align_op.time_shift, align_op_3.time_shift)

        out_1 = align_op({'image': image_2}, {'image': image_1})
        out_2 = align_op_3({'image': image_2}, {'image': image_1})
        self.assertTrue((out_1['image'] == out_2['image']).all())


if __name__ == '__main__':
    unittest.main()
