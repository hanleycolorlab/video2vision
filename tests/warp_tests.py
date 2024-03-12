import json
import unittest

import cv2
import numpy as np

import video2vision as v2v


class RotateTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This should be a 90 degree rotation, around the center of the image.
        warp_op = v2v.Rotate(90, output_size=(30, 20))

        # Check basic output
        image = np.arange(600., dtype=np.float32).reshape(20, 30)
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
        warp_op = v2v.Rotate(90, output_size=(200, 300))
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(200), np.arange(300))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        should_be = np.stack((ys - 50, 250 - xs), axis=1)
        self.assertTrue(
            (should_be == warp_op.apply_points(xys)).all(),
            (xys, should_be, warp_op.apply_points(xys)),
        )

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


class TranslationTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This should be a one point shift left
        warp_op = v2v.Translation(shift_x=-1, shift_y=0, output_size=(30, 20))

        # Check basic output
        image = np.arange(600., dtype=np.float32).reshape(20, 30)
        should_be = np.concatenate((image[:, 1:], np.zeros((20, 1))), axis=1)
        should_be = should_be.reshape(20, 30, 1).astype(np.float32)
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
        image = np.arange(600., dtype=np.float32).reshape(20, 30)
        should_be = np.concatenate(
            (np.zeros((20, 5)), image[:, 6:-4].T[::-1, :], np.zeros((20, 5))),
            axis=1
        )
        should_be = should_be.reshape(20, 30, 1).astype(np.float32)
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
        image = np.arange(600., dtype=np.float32).reshape(20, 30)
        should_be = np.concatenate(
            (np.zeros((20, 5)), image[:, 6:-4].T[::-1, :], np.zeros((20, 5))),
            axis=1
        )
        should_be = should_be.reshape(20, 30, 1).astype(np.float32)
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
