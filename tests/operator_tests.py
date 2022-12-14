'''
Note: when writing tests, make sure that the height and width of the images are
different, to detect height-width flip-flops.
'''
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

import video2vision as v2v


class ConcatenateOnBandsTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # Test image
        image_1 = np.zeros((4, 5, 3), dtype=np.float32)
        image_1[..., 1], image_1[..., 2] = 1, 2
        image_2 = np.zeros((4, 5, 3), dtype=np.float32)
        image_2[..., 0], image_2[..., 1], image_2[..., 2] = 3, 4, 5
        cat_op = v2v.ConcatenateOnBands(bands=[[0, 1], [1, 2]])
        dict_1 = {'image': image_1, 'dummy': 2, 'names': ['0']}
        dict_2 = {'image': image_2, 'dummy': 2, 'names': ['1']}
        cat_dict = cat_op(dict_1, dict_2)
        self.assertTrue(isinstance(cat_dict, dict))
        self.assertEqual(cat_dict.keys(), {'image', 'dummy', 'names'})
        self.assertEqual(cat_dict.get('dummy', 0), 2)
        self.assertEqual(cat_dict['names'], ['0-1'])
        cat_image = cat_dict['image']
        self.assertEqual(cat_image.shape, (4, 5, 4))
        self.assertEqual(cat_image.dtype, np.float32)
        should_be = np.array([0, 1, 4, 5]).reshape(1, 1, 4)
        self.assertTrue((cat_image == should_be).all())

        # Test video
        video_1 = np.zeros((4, 5, 3, 3), dtype=np.float32)
        video_1[..., 1], video_1[..., 2] = 1, 2
        video_2 = np.zeros((4, 5, 3, 3), dtype=np.float32)
        video_2[..., 0], video_2[..., 1], video_2[..., 2] = 3, 4, 5
        cat_dict = cat_op({'image': video_1}, {'image': video_2})
        self.assertTrue(isinstance(cat_dict, dict))
        self.assertEqual(cat_dict.keys(), {'image'})
        cat_video = cat_dict['image']
        self.assertEqual(cat_video.shape, (4, 5, 3, 4))
        self.assertEqual(cat_video.dtype, np.float32)
        should_be = np.array([0, 1, 4, 5]).reshape(1, 1, 1, 4)
        self.assertTrue((cat_video == should_be).all())

        # Test with one mask
        mask_1 = np.ones((4, 5), dtype=np.uint8)
        mask_1[2, 3] = 0
        cat_dict = cat_op(
            {'image': image_1, 'mask': mask_1},
            {'image': image_2}
        )
        self.assertTrue(isinstance(cat_dict, dict))
        self.assertEqual(cat_dict.keys(), {'image', 'mask'})
        self.assertEqual(cat_dict['mask'].shape, (4, 5))
        self.assertEqual(cat_dict['mask'].dtype, np.uint8)
        self.assertTrue((cat_dict['mask'] == mask_1).all())

        # Test with two masks
        mask_2 = np.ones((4, 5), dtype=np.uint8)
        mask_2[3, 1] = 0
        mask_should_be = np.ones((4, 5), dtype=np.uint8)
        mask_should_be[3, 1] = mask_should_be[2, 3] = 0
        cat_dict = cat_op(
            {'image': image_1, 'mask': mask_1},
            {'image': image_2, 'mask': mask_2}
        )
        self.assertTrue(isinstance(cat_dict, dict))
        self.assertEqual(cat_dict.keys(), {'image', 'mask'})
        self.assertEqual(cat_dict['mask'].shape, (4, 5))
        self.assertEqual(cat_dict['mask'].dtype, np.uint8)
        self.assertTrue((cat_dict['mask'] == mask_should_be).all())

        # Test handling wrong number of inputs
        with self.assertRaises(ValueError):
            cat_op({'image': image_1})
        # Test handling inputs with wrong number of bands
        with self.assertRaises(ValueError):
            cat_op({'image': image_1}, {'image': image_2[..., :2]})

        # Test handling inputs with disagreeing values
        with self.assertRaises(ValueError):
            cat_op(
                {'image': image_1, 'dummy': 1},
                {'image': image_2, 'dummy': 2},
            )

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        cat_op = v2v.ConcatenateOnBands(bands=[[0, 1], [1, 2]])
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == cat_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        cat_op = v2v.ConcatenateOnBands(bands=[[0, 1], [1, 2]])

        op_dict = json.loads(json.dumps(cat_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image_1 = np.zeros((4, 5, 3), dtype=np.float32)
        image_1[..., 1], image_1[..., 2] = 1, 2
        image_2 = np.zeros((4, 5, 3), dtype=np.float32)
        image_2[..., 0], image_2[..., 1], image_2[..., 2] = 3, 4, 5
        orig_op_out = cat_op({'image': image_1}, {'image': image_2})
        rest_op_out = restored_op({'image': image_1}, {'image': image_2})

        self.assertTrue((orig_op_out['image'] == rest_op_out['image']).all())


class LinearMapTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        lm_op = v2v.LinearMap(np.random.normal(0, 1, (3, 4)))

        # Check basic case
        image = np.arange(1800).reshape(30, 20, 3)
        out = lm_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        out = out['image']
        self.assertEqual(out.shape, (30, 20, 4))
        self.assertEqual(out.dtype, np.float32)

        # Check video
        image = np.arange(10 * 1800).reshape(30, 20, 10, 3)
        out = lm_op({'image': image})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        out = out['image']
        self.assertEqual(out.shape, (30, 20, 10, 4))
        self.assertEqual(out.dtype, np.float32)

        # Check can handle masks
        image = np.arange(1800, dtype=np.float32).reshape(30, 20, 3)
        mask = np.ones((30, 20), dtype=np.uint8)
        mask[:, 10] = 0
        out = lm_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertEqual(out['image'].shape, (30, 20, 4))
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].shape, (30, 20))
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertTrue((out['mask'] == mask).all())

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            lm_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            lm_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

        # Check handling of wrong number of bands
        with self.assertRaises(ValueError):
            lm_op({'image': np.arange(8).reshape(2, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        lm_op = v2v.LinearMap(np.random.normal(0, 1, (3, 4)))
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == lm_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        lm_op = v2v.LinearMap(np.random.normal(0, 1, (3, 4)))

        op_dict = json.loads(json.dumps(lm_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 3))
        orig_op_image = lm_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())

    def test_build_sensor_converter(self):
        '''
        Tests building a sensor converter.
        '''
        def _load_csv(path):
            return np.genfromtxt(path, skip_header=1, delimiter=',')[:, 1:]

        data_root = os.path.join(os.path.dirname(__file__), 'data')
        cam_illum_path = os.path.join(data_root, 'camera_illumination.csv')
        cam_illum = _load_csv(cam_illum_path)
        data_root = os.path.join(os.path.dirname(__file__), '../data')
        cam_sense_path = os.path.join(data_root, 'camera_sensitivities.csv')
        cam_sense = _load_csv(cam_sense_path)
        apis_sense_path = os.path.join(data_root, 'apis_sensitivities.csv')
        apis_sense = _load_csv(apis_sense_path)
        ref_path = os.path.join(data_root, 'material_reflectances.csv')
        reflectances = _load_csv(ref_path)
        ref_train, ref_test = reflectances[:, :-250], reflectances[:, -250:]

        lm_op = v2v.LinearMap.build_sensor_convertor(
            ref_train,
            cam_sense,
            apis_sense,
            source_illum=cam_illum,
        )

        cam_qc_test = ref_test.T.dot(cam_illum.reshape(-1, 1) * cam_sense)
        apis_qc_test = ref_test.T.dot(apis_sense)

        cam_image = cam_qc_test.reshape(10, 25, 4)
        apis_image = apis_qc_test.reshape(10, 25, 3)
        est_apis_image = lm_op({'image': cam_image})['image']

        self.assertTrue(np.abs(apis_image - est_apis_image).mean() < 1e-2)


class LoadOperatorTest(unittest.TestCase):
    def test_handling(self):
        lm_op = v2v.LinearMap(np.random.normal(0, 1, (3, 4)))
        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'operator.json')
            with open(path, 'w') as temp_file:
                json.dump(lm_op._to_json(), temp_file)
            new_op = v2v.load_operator(path)
        self.assertEqual(lm_op._to_json(), new_op._to_json())


class PadTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # Check basic polynomials
        image = np.arange(6.).reshape(2, 3)
        should_be = np.zeros((4, 5))
        should_be[1:3, 1:4] = image
        should_be = should_be.reshape(4, 5, 1)
        mask_should_be = np.zeros((4, 5), dtype=np.uint8)
        mask_should_be[1:3, 1:4] = 1

        pad_op = v2v.Pad((1, 1, 1, 1), 0)
        out = pad_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        out, mask = out['image'], out['mask']
        self.assertTrue((out == should_be).all())
        self.assertEqual(out.shape, (4, 5, 1))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((mask == mask_should_be).all(), mask)
        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(mask.dtype, np.uint8)

        # Check video
        video = np.stack([image.reshape(2, 3, 1)] * 7, axis=2)
        should_be = np.zeros((4, 5, 7))
        should_be[1:3, 1:4, :] = image.reshape(2, 3, 1)
        should_be = should_be.reshape(4, 5, 7, 1)
        out = pad_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        out, mask = out['image'], out['mask']
        self.assertTrue((out == should_be).all())
        self.assertTrue((mask == mask_should_be).all())
        self.assertEqual(mask.shape, (4, 5))

        # Check it can handle masks
        image = np.arange(6, dtype=np.float32).reshape(2, 3)
        should_be = np.zeros((4, 5), dtype=np.float32)
        should_be[1:3, 1:4] = image
        should_be = should_be.reshape(4, 5, 1)
        mask = np.ones((2, 3), dtype=np.uint8)
        mask[0, 2] = 0
        mask_should_be = np.ones((4, 5), dtype=np.uint8)
        mask_should_be[0, :] = mask_should_be[-1, :] = 0
        mask_should_be[:, 0] = mask_should_be[:, -1] = 0
        mask_should_be[1, 3] = 0
        out = pad_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertTrue(out['image'].shape, (4, 5, 1))
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertTrue((out['mask'] == mask_should_be).all())
        self.assertTrue(out['mask'].shape, (4, 5))
        self.assertEqual(out['mask'].dtype, np.uint8)

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            pad_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            pad_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        pad_op = v2v.Pad((1, 2, 3, 4), 0)
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        should_be = xys + np.array([[3, 1]])
        self.assertTrue((should_be == pad_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        pad_op = v2v.Pad((1, 1, 1, 1), 4)

        op_dict = json.loads(json.dumps(pad_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 3))
        orig_op_image = pad_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class ResizeTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # Test with float scale
        warp_op = v2v.Resize(scale=2., sampling_mode=cv2.INTER_NEAREST)

        # Check basic output
        image = np.ones((2, 3), dtype=np.float32)
        should_be = np.ones((4, 6, 1))
        out = warp_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (4, 6, 1))
        self.assertEqual(out['image'].dtype, np.float32)
        out = warp_op({'image': image.reshape(2, 3, 1)})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (4, 6, 1))

        # Check it can handle video
        video = np.stack([image.reshape(2, 3, 1)] * 4, axis=2)
        should_be = should_be.reshape(4, 6, 1, 1)
        out = warp_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['image'].shape, (4, 6, 4, 1))

        # Check that it can handle masks
        image = np.array([[1, 1], [1, 2]], dtype=np.float32) / 2.
        should_be = np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]]
        )
        should_be = should_be.reshape(4, 4, 1) / 2.
        mask = np.ones((2, 2), dtype=np.uint8)
        mask[1, 1] = 0
        mask_should_be = np.ones((4, 4), dtype=np.uint8)
        mask_should_be[2:4, 2:4] = 0
        out = warp_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((np.abs(out['image'] - should_be) < 1e-2).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertTrue((out['mask'] == mask_should_be).all())
        self.assertEqual(out['mask'].shape, (4, 4))
        self.assertEqual(out['mask'].dtype, np.uint8)

        # Test with ints scale
        warp_op = v2v.Resize(scale=(6, 4))

        # Check basic output
        image = np.ones((2, 3), dtype=np.float32)
        should_be = np.ones((4, 6, 1))
        out = warp_op({'image': image})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        self.assertTrue((out['image'] == should_be).all(), warp_op(image))
        self.assertEqual(out['image'].dtype, np.float32)
        out = warp_op({'image': image.reshape(2, 3, 1)})
        self.assertTrue((out['image'] == should_be).all())

        # Check it can handle video
        video = np.stack([image.reshape(2, 3, 1)] * 4, axis=2)
        should_be = should_be.reshape(4, 6, 1, 1)
        out = warp_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['image'].shape, (4, 6, 4, 1))

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        warp_op = v2v.Resize(scale=2.)
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue(((2 * xys) == warp_op.apply_points(xys)).all())

        warp_op = v2v.Resize(scale=(3, 4))
        with self.assertRaises(RuntimeError):
            warp_op.apply_points(xys)

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        warp_op = v2v.Resize(scale=2.)

        op_dict = json.loads(json.dumps(warp_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 2))
        orig_op_image = warp_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class RotateTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This should be a 90 degree rotation.
        warp_op = v2v.Rotate(90, output_size=(30, 20), center=(15, 10))

        # Check basic output
        image = np.arange(600.).reshape(20, 30)
        should_be = np.concatenate(
            (np.zeros((20, 5)), image[:, 6:-4].T[::-1, :], np.zeros((20, 5))),
            axis=1
        )
        should_be = should_be.reshape(20, 30, 1)
        mask_should_be = np.concatenate(
            (np.zeros((20, 5)), np.ones((20, 20)), np.zeros((20, 5))),
            axis=1
        )
        mask_should_be = mask_should_be.astype(np.uint8)
        out = warp_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['image'].shape, (20, 30, 1))
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check it can handle video
        video = np.stack([image.reshape(20, 30, 1)] * 4, axis=2)
        should_be = should_be.reshape(20, 30, 1, 1)
        out = warp_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (20, 30, 4, 1))
        self.assertEqual(out['image'].dtype, np.float32)

        # Check it can handle masks
        image = np.arange(600, dtype=np.float32).reshape(20, 30)
        should_be = should_be.reshape(20, 30, 1).astype(np.float32)
        mask = np.ones((20, 30), dtype=np.uint8)
        mask[0, 10] = 0
        mask_should_be[15, 5] = 0
        out = warp_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertTrue((out['mask'] == mask_should_be).all())
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        warp_op = v2v.Rotate(90, output_size=(200, 300), center=(150, 100))
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        should_be = np.stack((ys + 50, 250 - xs), axis=1)
        self.assertTrue((should_be == warp_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        warp_op = v2v.Rotate(angle=90, output_size=(30, 30))

        op_dict = json.loads(json.dumps(warp_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 2))
        orig_op_image = warp_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class TemporalShiftTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        with self.assertRaises(ValueError):
            v2v.TemporalShift(-1)

        shift_op = v2v.TemporalShift(2)

        with self.assertRaises(ValueError):
            shift_op({'image': np.arange(48).reshape(4, 4, 3)})
        with self.assertRaises(ValueError):
            shift_op({'image': np.arange(48).reshape(4, 4, 2, 3)})

        video_1 = np.random.normal(0, 1, (4, 4, 3, 1)).astype(np.float32)
        video_2 = np.random.normal(0, 1, (4, 4, 3, 1)).astype(np.float32)

        out_1 = shift_op({'image': video_1, 'dummy': 1})
        out_2 = shift_op({'image': video_2})

        self.assertEqual(out_1.get('dummy', 0), 1)
        out_1, out_2 = out_1['image'], out_2['image']

        self.assertEqual(out_1.shape, video_1.shape)
        self.assertEqual(out_2.shape, video_2.shape)

        self.assertTrue((out_1[:, :, :2] == 0).all())
        self.assertTrue((out_1[:, :, 2:3] == video_1[:, :, 0:1]).all())
        self.assertTrue((out_2[:, :, 0:2] == video_1[:, :, 1:3]).all())
        self.assertTrue((out_2[:, :, 2:3] == video_2[:, :, 0:1]).all())

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        shift_op = v2v.TemporalShift(1)
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == shift_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        shift_op = v2v.TemporalShift(1)

        op_dict = json.loads(json.dumps(shift_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 3, 1))
        orig_op_image = shift_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class TranslationTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This should be a one point shift left
        warp_op = v2v.Translation(shift_x=-1, shift_y=0, output_size=(30, 20))

        # Check basic output
        image = np.arange(600.).reshape(20, 30)
        should_be = np.concatenate((image[:, 1:], np.zeros((20, 1))), axis=1)
        should_be = should_be.reshape(20, 30, 1)
        mask_should_be = np.ones((20, 30), dtype=np.uint8)
        mask_should_be[:, -1] = 0
        out = warp_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['image'].shape, (20, 30, 1))
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check it can handle video
        video = np.stack([image.reshape(20, 30, 1)] * 4, axis=2)
        should_be = should_be.reshape(20, 30, 1, 1)
        out = warp_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (20, 30, 4, 1))
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check it can handle masks
        image = np.arange(600, dtype=np.float32).reshape(20, 30)
        mask = np.ones((20, 30), dtype=np.uint8)
        mask[10, 10] = 0
        should_be = np.concatenate(
            (image[:, 1:], np.zeros((20, 1), dtype=np.float32)), axis=1
        )
        should_be = should_be.reshape(20, 30, 1)
        mask_should_be[10, 9] = 0
        out = warp_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        warp_op = v2v.Translation(shift_x=24, shift_y=-9, output_size=(30, 30))
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        should_be = np.stack((xs + 24, ys - 9), axis=1)
        self.assertTrue((should_be == warp_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        warp_op = v2v.Translation(shift_x=3, shift_y=4, output_size=(30, 30))

        op_dict = json.loads(json.dumps(warp_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 2))
        orig_op_image = warp_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class WarpTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # First check warpAffine version

        # This should be a 90 degree rotation
        coe = cv2.getRotationMatrix2D((15, 10), 90, scale=1)
        warp_op = v2v.Warp(coe=coe, output_size=(30, 20))

        # Check basic output
        image = np.arange(600.).reshape(20, 30)
        should_be = np.concatenate(
            (np.zeros((20, 5)), image[:, 6:-4].T[::-1, :], np.zeros((20, 5))),
            axis=1
        )
        should_be = should_be.reshape(20, 30, 1)
        mask_should_be = np.concatenate(
            (np.zeros((20, 5)), np.ones((20, 20)), np.zeros((20, 5))),
            axis=1
        )
        mask_should_be = mask_should_be.astype(np.uint8)
        out = warp_op({'image': image, 'dummy': 1})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 1)
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['image'].shape, (20, 30, 1))
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check it can handle video
        video = np.stack([image.reshape(20, 30, 1)] * 4, axis=2)
        should_be = should_be.reshape(20, 30, 1, 1)
        out = warp_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (20, 30, 4, 1))
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Check that it can handle masks
        image = np.arange(600, dtype=np.float32).reshape(20, 30)
        should_be = should_be.reshape(20, 30, 1).astype(np.float32)
        mask_should_be[15, 5] = 0
        mask = np.ones((20, 30), dtype=np.uint8)
        mask[0, 10] = 0
        out = warp_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertEqual(out['image'].shape, (20, 30, 1))
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertEqual(out['mask'].shape, (20, 30))
        self.assertTrue((out['mask'] == mask_should_be).all())

        # Now check warpPerspective version

        # This should be a 90 degree rotation
        coe = cv2.getRotationMatrix2D((15, 10), 90, scale=1)
        coe = np.concatenate([coe, np.array([[0, 0, 1]])], axis=0)
        warp_op = v2v.Warp(coe=coe, output_size=(30, 20))

        # Check basic output
        image = np.arange(600.).reshape(20, 30)
        should_be = np.concatenate(
            (np.zeros((20, 5)), image[:, 6:-4].T[::-1, :], np.zeros((20, 5))),
            axis=1
        )
        should_be = should_be.reshape(20, 30, 1)
        out = warp_op({'image': image})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (20, 30, 1))
        video = np.stack([image.reshape(20, 30, 1)] * 4, axis=2)
        out = warp_op({'image': video})
        should_be = should_be.reshape(20, 30, 1, 1)
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (20, 30, 4, 1))
        self.assertEqual(out['image'].dtype, np.float32)

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            warp_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        coe = cv2.getRotationMatrix2D((150, 100), 90, scale=1)
        warp_op = v2v.Warp(coe=coe, output_size=(200, 300))
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        should_be = np.stack((ys + 50, 250 - xs), axis=1)
        self.assertTrue((should_be == warp_op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        coe = cv2.getRotationMatrix2D((14.5, 14.5), 90, scale=1)
        warp_op = v2v.Warp(coe=coe, output_size=(30, 30))

        op_dict = json.loads(json.dumps(warp_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 2))
        orig_op_image = warp_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())

    def test_building_from_tiepoints(self):
        # This should be a 90 degree rotation
        source_pts = np.array([[0., 0.], [0., 9.], [9., 9.], [9., 0.]])
        control_pts = np.array([[0., 9.], [9., 9.], [9., 0.], [0., 0.]])
        warp_op = v2v.Warp.build_from_tiepoints(
            source_pts, control_pts, (10, 10)
        )
        image = np.arange(100.).reshape(10, 10, 1)
        out = warp_op({'image': image})['image']
        should_be = image.reshape(10, 10).T[::-1, :].reshape(10, 10, 1)
        self.assertEqual(out.shape, (10, 10, 1))
        self.assertTrue((out == should_be).all(), out.reshape(10, 10))


if __name__ == '__main__':
    unittest.main()
