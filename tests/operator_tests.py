'''
Note: when writing tests, make sure that the height and width of the images are
different, to detect height-width flip-flops.
'''
import json
from math import sqrt
import os
import tempfile
import unittest

import cv2
import numpy as np

import video2vision as v2v


class BaseClassTests(unittest.TestCase):
    def test_hold_token(self):
        op = v2v.Operator()
        token = v2v.operators.HoldToken()
        self.assertTrue(isinstance(op(token), v2v.operators.HoldToken))


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
        apis_sense_path = os.path.join(
            data_root, 'animal_sensitivities/apis_sensitivities.csv'
        )
        apis_sense = _load_csv(apis_sense_path)
        ref_path = os.path.join(data_root, 'material_reflectances.csv')
        if not os.path.exists(ref_path):
            return
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


class ToRNLTest(unittest.TestCase):
    def test_handling(self):
        op = v2v.ToRNL(
            illuminance=1.,
            background=1.,
            photo_sensitivity=np.ones((1, 5), dtype=np.float32),
            photo_density=np.ones(5, dtype=np.float32),
            weber_fraction=1.
        )

        with self.assertRaises(NotImplementedError):
            op({'image': np.ones((1, 1, 5), dtype=np.float32)})

        # 3-channel test

        op = v2v.ToRNL(
            illuminance=1.,
            background=1.,
            photo_sensitivity=np.ones((1, 3), dtype=np.float32),
            photo_density=np.ones(3, dtype=np.float32),
            weber_fraction=1.
        )

        out = op({
            'image': np.ones((1, 1, 3), dtype=np.float32),
            'dummy': 1,
        })

        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'dummy', 'image'})
        self.assertTrue(out['dummy'], 1)
        self.assertTrue(isinstance(out['image'], np.ndarray))
        self.assertEqual(out['image'].shape, (1, 1, 2))
        self.assertTrue(np.isclose(out['image'], 0.4641016).all())

        # 4-channel test

        op = v2v.ToRNL(
            illuminance=1.,
            background=1.,
            photo_sensitivity=np.ones((1, 4), dtype=np.float32),
            photo_density=np.ones(4, dtype=np.float32),
            weber_fraction=1.
        )

        out = op({
            'image': np.ones((1, 1, 4), dtype=np.float32),
            'dummy': 1,
        })

        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'dummy', 'image'})
        self.assertTrue(out['dummy'], 1)
        self.assertTrue(isinstance(out['image'], np.ndarray))
        self.assertEqual(out['image'].shape, (1, 1, 3))
        self.assertTrue(np.isclose(out['image'], 0.4494897).all())

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        op = v2v.ToRNL(
            illuminance=1.,
            background=1.,
            photo_sensitivity=np.ones((1, 4), dtype=np.float32),
            photo_density=np.ones(4, dtype=np.float32),
            weber_fraction=1.
        )

        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        op = v2v.ToRNL(
            illuminance=1.,
            background=1.,
            photo_sensitivity=np.ones((1, 4), dtype=np.float32),
            photo_density=np.ones(4, dtype=np.float32),
            weber_fraction=1.
        )

        op_dict = json.loads(json.dumps(op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.uniform(0, 1, (4, 5, 4)).astype(np.float32)
        orig_op_out = op({'image': image})
        rest_op_out = restored_op({'image': image})

        self.assertTrue((orig_op_out['image'] == rest_op_out['image']).all())


class UBGRtoXYZTest(unittest.TestCase):
    def test_handling(self):
        op = v2v.UBGRtoXYZ()

        with self.assertRaises(ValueError):
            op({'image': np.arange(3, dtype=np.float32).reshape(1, 1, 3)})

        out = op({
            'image': np.arange(4, dtype=np.float32).reshape(1, 1, 4),
            'dummy': 1,
        })

        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'dummy', 'image'})
        self.assertTrue(out['dummy'], 1)
        self.assertTrue(isinstance(out['image'], np.ndarray))
        self.assertEqual(out['image'].shape, (1, 1, 3))
        out = out['image'].flatten()
        should_be = np.array([-0.25, 0., sqrt(3 / 2) / 6]).astype(np.float32)
        self.assertTrue(np.abs((out - should_be) < 1e-4).all())

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        op = v2v.UBGRtoXYZ()
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == op.apply_points(xys)).all())

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        op = v2v.UBGRtoXYZ()

        op_dict = json.loads(json.dumps(op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.uniform(0, 1, (4, 5, 4)).astype(np.float32)
        orig_op_out = op({'image': image})
        rest_op_out = restored_op({'image': image})

        self.assertTrue((orig_op_out['image'] == rest_op_out['image']).all())


if __name__ == '__main__':
    unittest.main()
