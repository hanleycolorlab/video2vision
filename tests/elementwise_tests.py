import json
from math import log
import unittest

import numpy as np

import video2vision as v2v


class PolynomialTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This will be y = 1 + x^2
        poly = np.polynomial.Polynomial((1, 0, 1))
        poly_op = v2v.Polynomial([poly])

        # Check basic polynomials
        image = np.arange(6., dtype=np.float32).reshape(2, 3)
        should_be = np.array([1, 2, 5, 10, 17, 26]).reshape(2, 3, 1)
        # Check dummy value is preserved
        out = poly_op({'image': image, 'dummy': 2})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 2)
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)

        # Check video
        video = np.stack([image.reshape(2, 3, 1)] * 4, axis=2)
        should_be = should_be.reshape(2, 3, 1, 1)
        out = poly_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].shape, (2, 3, 4, 1))

        # Check it can handle masks
        image = np.arange(6, dtype=np.float32).reshape(2, 3)
        should_be = np.array([1, 2, 5, 10, 17, 26]).reshape(2, 3, 1)
        mask = np.ones((2, 3), dtype=np.uint8)
        mask[1, 1] = 0
        out = poly_op({'image': image, 'mask': mask})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((out['image'] == should_be).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertTrue((out['mask'] == mask).all())
        self.assertEqual(out['mask'].shape, (2, 3))

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            poly_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            poly_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

        # Check handling of wrong number of bands
        with self.assertRaises(ValueError):
            poly_op({'image': np.arange(8).reshape(2, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        poly = np.polynomial.Polynomial((1, 0, 1))
        poly_op = v2v.Polynomial([poly])
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == poly_op.apply_points(xys)).all())

    def test_apply_values(self):
        # This will be y = 1 + x^2
        poly_0 = np.polynomial.Polynomial((1, 0, 1))
        # This will be y = 2x - 3
        poly_1 = np.polynomial.Polynomial((-3, 2, 0))
        poly_op = v2v.Polynomial([poly_0, poly_1])
        values = np.array([[1., 2.], [2., 1.], [0., 0.]])
        should_be = np.array([[2., 1.], [5., -1.], [1., -3.]])
        out = poly_op.apply_values(values)
        self.assertTrue((out == should_be).all(), out)

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        poly = np.polynomial.Polynomial((1, 0, 1))
        poly_op = v2v.Polynomial([poly])

        op_dict = json.loads(json.dumps(poly_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 1))
        orig_op_image = poly_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


class PowerLawTest(unittest.TestCase):
    def test_handling(self):
        '''
        This tests basic coercion and handling.
        '''
        # This will be y = 1.5 * 1.1^x - 1.
        power_op = v2v.PowerLaw([[1.5, 1.1, -1]])

        # Check basic outputs
        image = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
        should_be = np.array([[0.5, 0.65], [0.815, 0.9965]]).reshape(2, 2, 1)
        # Check extra variables are preserved
        out = power_op({'image': image, 'dummy': 2})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'dummy'})
        self.assertEqual(out.get('dummy', 0), 2)
        self.assertTrue((np.abs(out['image'] - should_be) < 1e-2).all())
        self.assertEqual(out['image'].dtype, np.float32)

        # Check video
        video = np.stack([image.reshape(2, 2, 1)] * 4, axis=2)
        should_be = should_be.reshape(2, 2, 1, 1)
        out = power_op({'image': video})
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image'})
        self.assertTrue((np.abs(out['image'] - should_be) < 1e-2).all())
        self.assertEqual(out['image'].shape, (2, 2, 4, 1))

        # Check it can handle masks
        mask = np.ones((2, 2), dtype=np.uint8)
        mask[1, 1] = 0
        out = power_op({'image': image, 'mask': mask})
        should_be = should_be.reshape(2, 2, 1)
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'mask'})
        self.assertTrue((np.abs(out['image'] - should_be) < 1e-2).all())
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertEqual(out['mask'].dtype, np.uint8)
        self.assertTrue((out['mask'] == mask).all())
        self.assertEqual(out['mask'].shape, (2, 2))

        # Check handling of wrong number of dimensions
        with self.assertRaises(ValueError):
            power_op({'image': np.arange(4)})
        with self.assertRaises(ValueError):
            power_op({'image': np.arange(16).reshape(2, 2, 1, 2, 2)})

        # Check handling of wrong number of bands
        with self.assertRaises(ValueError):
            power_op({'image': np.arange(8).reshape(2, 2, 2)})

    def test_apply_points(self):
        '''
        Tests the apply_points method.
        '''
        power_op = v2v.PowerLaw([[1.5, 1.01, -1]])
        # Test apply_points
        xs, ys = np.meshgrid(np.arange(300), np.arange(200))
        xs, ys = xs.flatten(), ys.flatten()
        xys = np.stack((xs, ys), axis=1)
        self.assertTrue((xys == power_op.apply_points(xys)).all())

    def test_apply_values(self):
        # This will be y0 = 1 + 2^x, y1 = 2 * 0.5^x
        power_op = v2v.PowerLaw([[1., 2., 1.], [2., 0.5, 0.0]])
        values = np.array([[1., 2.], [2., 1.], [0., 0.]])
        should_be = np.array([[3., 0.5], [5., 1.], [2., 2.]])
        out = power_op.apply_values(values)
        self.assertTrue((out == should_be).all(), out)

    def test_power_law_formula(self):
        # This will be y = 1 + 2^x
        power_law = v2v.elementwise.PowerLawFormula(1, 2, 1, 1)
        x = np.array([0, 1, 2], dtype=np.float32)
        out = np.zeros_like(x)
        y = power_law(x, out=out)
        self.assertTrue(y is out)
        should_be = np.array([3 - (2 * log(2)), 3, 5])
        self.assertTrue((np.abs(y - should_be) < 1e-4).all(), y)

    def test_serialization(self):
        '''
        Tests serialization to disk and back.
        '''
        power_op = v2v.PowerLaw([[1.5, 1.01, -1]])

        op_dict = json.loads(json.dumps(power_op._to_json()))
        op_class = v2v.OPERATOR_REGISTRY.get(op_dict.pop('class'))
        restored_op = op_class(**op_dict)

        image = np.random.normal(0, 1, (30, 30, 1))
        orig_op_image = power_op({'image': image})['image']
        rest_op_image = restored_op({'image': image})['image']

        self.assertTrue((orig_op_image == rest_op_image).all())


if __name__ == '__main__':
    unittest.main()
