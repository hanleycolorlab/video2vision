import os
import tempfile
from typing import Dict
import unittest
import warnings

import numpy as np

import video2vision as v2v


class ResetCounterOperator(v2v.Operator):
    def __init__(self):
        self.num_resets = 0

    def apply(self, x: Dict) -> Dict:
        return x

    def reset(self):
        self.num_resets += 1


class ResetTriggerOperator(ResetCounterOperator):
    def apply(self, x: Dict):
        if self.num_resets == 0:
            raise v2v.ResetPipeline()
        return x


class PipelineTest(unittest.TestCase):
    def _make_images(self, temp_root: str):
        # Create input directory
        in_root = os.path.join(temp_root, 'input')
        os.makedirs(in_root)

        temp_path = os.path.join(in_root, '1.png')
        self.image_1 = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_1[:2, :2] = (255, 255, 0)
        v2v.save(self.image_1, temp_path)

        temp_path = os.path.join(in_root, '2.png')
        self.image_2 = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image_2[2:4, 2:4] = (0, 255, 255)
        v2v.save(self.image_2, temp_path)

        # Create output directories
        out_root_1 = os.path.join(temp_root, 'out_1')
        os.makedirs(out_root_1)
        out_root_2 = os.path.join(temp_root, 'out_2')
        os.makedirs(out_root_2)

        return in_root, out_root_1, out_root_2

    def _make_pipeline(self, temp_root: str, batch_size: int):
        in_root, out_root_1, out_root_2 = self._make_images(temp_root)

        pipe = v2v.Pipeline()

        # Create loader
        loader_idx = pipe.add_operator(v2v.Loader(in_root, (8, 8), batch_size))

        # Create 90 degree rotater, and connect loader to it
        rot_idx = pipe.add_operator(v2v.Rotate(90, (8, 8), center=(3.5, 3.5)))
        pipe.add_edge(loader_idx, rot_idx, in_slot=0)

        # Create writer, and connect rotator to it
        writer_1_idx = pipe.add_operator(v2v.Writer(out_root_1))
        pipe.add_edge(rot_idx, writer_1_idx, in_slot=0)

        # Create translator, and connect rotator to it
        trans_idx = pipe.add_operator(v2v.Translation(0, 1, (8, 8)))
        pipe.add_edge(rot_idx, trans_idx, in_slot=0)

        # Create writer, and connect translator to it
        writer_2_idx = pipe.add_operator(v2v.Writer(out_root_2))
        pipe.add_edge(trans_idx, writer_2_idx, in_slot=0)

        return pipe

    def _check_output_1(self, temp_path: str):
        if isinstance(temp_path, str):
            out_root_1 = os.path.join(temp_path, 'out_1')
            rot_path_1 = os.path.join(out_root_1, '1.tif')
            rtn_1 = v2v.load(rot_path_1)
            rot_path_2 = os.path.join(out_root_1, '2.tif')
            rtn_2 = v2v.load(rot_path_2)
        else:
            rtn_1, rtn_2 = temp_path

        self.assertTrue(rtn_1.shape, (8, 8, 3))
        rot_should_be_1 = np.rollaxis(self.image_1, 1, 0)[::-1]
        self.assertTrue((rtn_1 == rot_should_be_1).all())

        self.assertTrue(rtn_2.shape, (8, 8, 3))
        rot_should_be_2 = np.rollaxis(self.image_2, 1, 0)[::-1]
        self.assertTrue((rtn_2 == rot_should_be_2).all())

    def _check_output_2(self, temp_path: str):
        if isinstance(temp_path, str):
            out_root_2 = os.path.join(temp_path, 'out_2')
            trans_path_1 = os.path.join(out_root_2, '1.tif')
            rtn_1 = v2v.load(trans_path_1)
            trans_path_2 = os.path.join(out_root_2, '2.tif')
            rtn_2 = v2v.load(trans_path_2)
        else:
            rtn_1, rtn_2 = temp_path

        self.assertTrue(rtn_1.shape, (8, 8, 3))
        zeros = np.zeros((1, 8, 3), dtype=np.uint8)
        rot_should_be_1 = np.rollaxis(self.image_1, 1, 0)[::-1]
        trans_should_be_1 = np.concatenate(
            (zeros, rot_should_be_1[:-1]), axis=0
        )
        self.assertTrue((rtn_1 == trans_should_be_1).all())

        self.assertTrue(rtn_2.shape, (8, 8, 3))
        rot_should_be_2 = np.rollaxis(self.image_2, 1, 0)[::-1]
        trans_should_be_2 = np.concatenate(
            (zeros, rot_should_be_2[:-1]), axis=0
        )
        self.assertTrue((rtn_2 == trans_should_be_2).all())

    def _check_outputs(self, temp_path: str):
        self._check_output_1(temp_path)
        self._check_output_2(temp_path)

    def test_batch_size_1(self):
        '''
        Tests basic handling with batch_size=1.
        '''
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=1)

            # Run the pipeline
            pipe.run()

            self._check_outputs(temp_path)

    def test_batch_size_3(self):
        '''
        Tests basic handling with batch_size=3.
        '''
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=3)

            # Run the pipeline
            pipe.run()

            self._check_outputs(temp_path)

    def test_call(self):
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=1)
            in_path_1 = os.path.join(temp_path, 'input/1.png')
            image_1 = v2v.load(in_path_1)
            in_path_2 = os.path.join(temp_path, 'input/2.png')
            image_2 = v2v.load(in_path_2)

            out_a_1, out_a_2 = pipe(image_1)
            out_b_1, out_b_2 = pipe(image_2)

            self._check_output_1((out_a_1, out_b_1))
            self._check_output_2((out_a_2, out_b_2))

    def test_chain(self):
        '''
        Tests the :meth:`chain` method.
        '''
        with tempfile.TemporaryDirectory() as temp_path:
            in_root, _, out_root_2 = self._make_images(temp_path)
            # This may raise a harmless ImportWarning if pandas is not
            # installed. We suppress the warning.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ImportWarning)
                pipe = v2v.Pipeline.chain(
                    v2v.Loader(in_root, (8, 8)),
                    v2v.Rotate(90, (8, 8), center=(3.5, 3.5)),
                    v2v.Translation(0, 1, (8, 8)),
                    v2v.Writer(out_root_2),
                )
            pipe.run()
            self._check_output_2(temp_path)

    def test_full_up(self):
        '''
        This is a full-up test, to ensure we can replicate previous work.
        '''
        data_root = os.path.join(os.path.dirname(__file__), 'data')
        pipe = v2v.load_pipeline(os.path.join(data_root, 'pipeline.json'))
        pipe.set_loader_paths(os.path.join(data_root, 'uv_sample.jpg'))

        with tempfile.TemporaryDirectory() as temp_path:
            pipe.set_writer_paths(temp_path)
            pipe.run()
            out_path = os.path.join(temp_path, 'uv_sample.jpg')
            self.assertTrue(os.path.exists(out_path))
            out = v2v.load(out_path)

        should_be_path = os.path.join(data_root, 'uv_aligned_sample.jpg')
        should_be = v2v.load(should_be_path)

        self.assertEqual(out.shape, should_be.shape)
        self.assertEqual(out.dtype, should_be.dtype)
        self.assertTrue((np.abs(out - should_be) < 0.03).all())

    def test_serialization(self):
        '''
        Tests pipeline serialization.
        '''
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=1)

            pipe_path = os.path.join(temp_path, 'pipeline.json')
            pipe.save(pipe_path)
            pipe = v2v.Pipeline.load(pipe_path)

            in_root = os.path.join(temp_path, 'input')
            pipe.set_loader_paths(in_root)
            out_root_1 = os.path.join(temp_path, 'out_1')
            out_root_2 = os.path.join(temp_path, 'out_2')
            pipe.set_writer_paths(out_root_1, out_root_2)

            # Run the pipeline
            pipe.run()

            self._check_outputs(temp_path)

    def test_reset_all_inputs(self):
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=1)
            pipe._reset_all_inputs()
            pipe.nodes[1]['inputs'][0] = {'image': 'test'}
            pipe._reset_all_inputs()
            self.assertTrue(all(x is None for x in pipe.nodes[1]['inputs']))

    def test_reset_input(self):
        with tempfile.TemporaryDirectory() as temp_path:
            pipe = self._make_pipeline(temp_path, batch_size=1)
            pipe._reset_all_inputs()
            pipe.nodes[1]['inputs'][0] = {'image': 'test'}
            pipe._reset_inputs(pipe.nodes[1])
            self.assertTrue(all(x is None for x in pipe.nodes[1]['inputs']))

    def test_reset_operators(self):
        op = ResetCounterOperator()
        pipe = v2v.Pipeline()
        pipe.add_operator(op, idx=0)
        self.assertEqual(op.num_resets, 0)
        pipe._reset_all_operators()
        self.assertEqual(op.num_resets, 1)

    def test_raise_resetpipeline_exception(self):
        with tempfile.TemporaryDirectory() as temp_path:
            in_root, _, _ = self._make_images(temp_path)
            pipe = v2v.Pipeline()
            loader_idx = pipe.add_operator(v2v.Loader(in_root, (8, 8), 1))
            counter = ResetCounterOperator()
            count_idx = pipe.add_operator(counter)
            pipe.add_edge(loader_idx, count_idx, in_slot=0)
            trigger = ResetTriggerOperator()
            reset_idx = pipe.add_operator(trigger)
            pipe.add_edge(count_idx, reset_idx, in_slot=0)

            self.assertEqual(counter.num_resets, 0)
            self.assertEqual(trigger.num_resets, 0)

            pipe.run()

            self.assertEqual(counter.num_resets, 1)
            self.assertEqual(trigger.num_resets, 1)


if __name__ == '__main__':
    unittest.main()
