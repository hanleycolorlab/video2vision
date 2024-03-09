import os
import tempfile
import unittest

import numpy as np

import video2vision as v2v


def _is_close(x: np.ndarray, y: np.ndarray) -> bool:
    '''
    Convenience function used to compare two values under lossy compression.
    '''
    x, y = x.astype(np.float32), y.astype(np.float32)
    return (np.abs(x - y).mean() < 3)


def _get_video() -> np.ndarray:
    x = np.zeros((256, 256, 3, 3), dtype=np.float32)
    x[:128, :128, 0] = x[128:, :128, 1] = x[128:, 128:, 2] = (0, 0, 1)
    x[128:, :128, 0] = x[128:, 128:, 1] = x[:128, 128:, 2] = (0, 1, 0)
    x[128:, 128:, 0] = x[:128, 128:, 1] = x[:128, :128, 2] = (1, 0, 0)
    return x


class IOTest(unittest.TestCase):
    def test_load_arw(self):
        '''
        Tests the ability of :func:`load` to load arw files.
        '''
        path = os.path.join(os.path.dirname(__file__), 'data/raw_example.arw')
        image = v2v.load(path)
        should_be = np.array([456, 456, 474, 492, 493, 494, 494, 494.]) / 15360
        self.assertTrue((np.abs(image[:8, 0, 2] - should_be) < 1e-7).all())

        out = np.empty_like(image)
        image = v2v.load(path, out=out)
        self.assertTrue((np.abs(image[:8, 0, 2] - should_be) < 1e-7).all())
        self.assertTrue((np.abs(out[:8, 0, 2] - should_be) < 1e-7).all())

        image = v2v.load(path, noscale=True)
        should_be = np.array([456, 456, 474, 492, 493, 494, 494, 494])
        self.assertTrue((np.abs(image[:8, 0, 2] - should_be) < 1e-7).all())

    def test_load_and_save(self):
        '''
        Tests the :func:`load` and :func:`save` functions.
        '''
        # Note: don't use lossy formats here.

        # Can't use arw here because rawpy does not support saving arws.
        TEST_CASES = {
            'png': [
                (8, 8, 1),
                (8, 8, 3),
            ],
            'tif': [
                (8, 8, 1),
                (8, 8, 3),
                (8, 8, 6),
            ],
            # Although save can write grayscale fine, it will be converted to
            # BGR on read by cv2, and this cannot be avoided as per:
            #
            # https://stackoverflow.com/questions/11159506/opencv-videocapture-set-greyscale
            #
            # Not yet sure if it's SAVED as grayscale but just cast to BGR on
            # load, or if it's converted to BGR on save.
            'mp4': [
                (8, 8, 4, 3),
            ]
        }

        with tempfile.TemporaryDirectory() as temp_root:
            for ext, cases in TEST_CASES.items():
                temp_path = os.path.join(temp_root, f'test.{ext}')
                for shape in cases:
                    image = np.zeros(shape, dtype=np.float32)
                    image[4:, :4], image[:4, 4:], image[4:, 4:] = 0.2, 0.5, 0.7
                    v2v.save(image, temp_path)
                    rtn = v2v.load(temp_path)
                    self.assertEqual(rtn.dtype, np.float32, (ext, shape))
                    self.assertEqual(rtn.shape, shape, (ext, shape))
                    self.assertTrue(_is_close(rtn, image))

                    out = np.empty_like(image)
                    v2v.save(image, temp_path)
                    rtn = v2v.load(temp_path, out=out)
                    self.assertEqual(rtn.dtype, np.float32, (ext, shape))
                    self.assertEqual(rtn.shape, shape, (ext, shape))
                    self.assertTrue(_is_close(rtn, image))
                    self.assertTrue(_is_close(rtn, out))

    def test_load_noscale(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_path = os.path.join(temp_root, 'test.png')
            image = np.zeros((8, 8, 3), dtype=np.float32)
            image[4:, :4], image[:4, 4:], image[4:, 4:] = 0.2, 0.5, 0.7
            v2v.save(image, temp_path)

            rtn = v2v.load(temp_path, noscale=True)
            self.assertEqual(rtn.dtype, np.uint8)
            self.assertEqual(rtn.shape, (8, 8, 3))
            self.assertTrue(_is_close(rtn, (256 * image)))

    def test_load_error_handling(self):
        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.png')
            with open(path, 'w') as out_file:
                out_file.write('Dummy')

            with self.assertRaises(RuntimeError):
                v2v.load(path)

    def test_loader_handling(self):
        '''
        Tests basic handling of the :class:`Loader`.
        '''
        image_1 = np.zeros((16, 16, 3), dtype=np.float32)
        image_2 = np.full((16, 16, 3), 255. / 256., dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_root:
            temp_path_1 = os.path.join(temp_root, '1.png')
            v2v.save(image_1, temp_path_1)
            temp_path_2 = os.path.join(temp_root, '2.png')
            v2v.save(image_2, temp_path_2)
            loader = v2v.Loader(temp_root, (16, 16))
            self.assertEqual(len(loader), 2)

            # Test with batch_size = 1
            out_1 = loader()
            self.assertTrue(isinstance(out_1, dict))
            self.assertEqual(out_1.keys(), {'image', 'names', 'final'})
            self.assertEqual(out_1['names'], ['1'])
            self.assertEqual(out_1['image'].shape, (16, 16, 1, 3))
            self.assertEqual(out_1['image'].dtype, np.float32)
            self.assertTrue((out_1['image'] == 0).all())
            self.assertFalse(out_1['final'])

            out_2 = loader()
            self.assertTrue(isinstance(out_2, dict))
            self.assertEqual(out_2.keys(), {'image', 'names', 'final'})
            self.assertEqual(out_2['names'], ['2'])
            self.assertEqual(out_2['image'].shape, (16, 16, 1, 3))
            self.assertEqual(out_2['image'].dtype, np.float32)
            self.assertTrue((out_2['image'] == (255. / 256.)).all())
            self.assertTrue(out_2['final'])

            with self.assertRaises(v2v.OutOfInputs):
                loader()

            # Test with batch_size = 3
            loader = v2v.Loader(temp_root, (16, 16), batch_size=3)
            self.assertEqual(len(loader), 2)

            out = loader()
            self.assertTrue(isinstance(out, dict))
            self.assertEqual(out.keys(), {'image', 'names', 'final'})
            self.assertEqual(out['names'], ['1', '2'])
            self.assertEqual(out['image'].shape, (16, 16, 2, 3))
            self.assertEqual(out['image'].dtype, np.float32)
            self.assertTrue((out['image'][:, :, 0, :] == image_1).all())
            self.assertTrue((out['image'][:, :, 1, :] == image_2).all())
            self.assertTrue(out['final'])

            with self.assertRaises(v2v.OutOfInputs):
                loader()

            # Test passing the Loader a List[str]
            loader.set_path([temp_path_1, temp_path_2])
            out = loader()
            self.assertTrue(isinstance(out, dict))
            self.assertEqual(out.keys(), {'image', 'names', 'final'})
            self.assertEqual(out['names'], ['1', '2'])
            self.assertEqual(out['image'].shape, (16, 16, 2, 3))
            self.assertEqual(out['image'].dtype, np.float32)
            self.assertTrue((out['image'][:, :, 0, :] == image_1).all())
            self.assertTrue((out['image'][:, :, 1, :] == image_2).all())
            self.assertTrue(out['final'])

    def test_iter_check_size_with_png(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_path_1 = os.path.join(temp_root, '1.png')
            v2v.save(np.zeros((16, 16, 3), dtype=np.float32), temp_path_1)
            temp_path_2 = os.path.join(temp_root, '2.png')
            v2v.save(np.zeros((17, 16, 3), dtype=np.float32), temp_path_2)
            loader = v2v.Loader(
                [temp_path_1, temp_path_2],
                expected_size=(16, 16),
            )
            loader()
            with self.assertRaises(RuntimeError):
                loader()

    def test_iter_check_size_with_mp4(self):
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            v2v.save(video, path)
            loader = v2v.Loader(temp_root, (8, 8), batch_size=1)

            with self.assertRaises(RuntimeError):
                loader()

    def test_get_frame_check_size_with_png(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_path_1 = os.path.join(temp_root, '1.png')
            v2v.save(np.zeros((16, 16, 3), dtype=np.float32), temp_path_1)
            temp_path_2 = os.path.join(temp_root, '2.png')
            v2v.save(np.zeros((17, 16, 3), dtype=np.float32), temp_path_2)
            loader = v2v.Loader(
                [temp_path_1, temp_path_2],
                expected_size=(16, 16)
            )
            loader.get_frame(0)
            with self.assertRaises(RuntimeError):
                loader.get_frame(1)

    def test_get_frame_check_size_with_mp4(self):
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            v2v.save(video, path)
            loader = v2v.Loader(temp_root, batch_size=1, expected_size=(8, 8))

            with self.assertRaises(RuntimeError):
                loader.get_frame(0)

    def test_loader_buffer(self):
        '''
        Tests the :class:`Loader`'s ability to work with internal buffers.
        '''
        image_1 = np.zeros((16, 16, 3), dtype=np.float32)
        image_2 = np.full((16, 16, 3), 255. / 256., dtype=np.float32)
        loader = v2v.Loader(None, (16, 16), batch_size=2)

        # Test for error before we actually add the buffer
        with self.assertRaises(v2v.io.OutOfInputs):
            with v2v.io._read_write_from_to_buffer():
                loader()

        loader.buff = np.stack((image_1, image_2), axis=0)

        with v2v.io._read_write_from_to_buffer():
            out = loader()

        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out.keys(), {'image', 'final'})
        self.assertEqual(out['image'].shape, (16, 16, 2, 3))
        self.assertEqual(out['image'].dtype, np.float32)
        self.assertTrue((out['image'][:, :, 0, :] == image_1).all())
        self.assertTrue((out['image'][:, :, 1, :] == image_2).all())
        self.assertTrue(out['final'])
        self.assertEqual(loader.buff, None)

    def test_loader_video_handling(self):
        '''
        Tests the :class:`Loader`'s ability to handle video.
        '''
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            v2v.save(video, path)

            loader = v2v.Loader(temp_root, (256, 256), batch_size=2)
            out = loader()
            self.assertTrue(isinstance(out, dict))
            self.assertEqual(out.keys(), {'image', 'names', 'final'})
            self.assertEqual(out['names'], ['test', 'test'])
            self.assertEqual(out['image'].shape, (256, 256, 2, 3))
            self.assertEqual(out['image'].dtype, np.float32)
            self.assertTrue(_is_close(out['image'], video[:, :, :2, :]))
            self.assertFalse(out['final'])

            out = loader()
            self.assertTrue(isinstance(out, dict))
            self.assertEqual(out.keys(), {'image', 'names', 'final'})
            self.assertEqual(out['names'], ['test'])
            self.assertEqual(out['image'].shape, (256, 256, 1, 3))
            self.assertEqual(out['image'].dtype, np.float32)
            self.assertTrue(_is_close(out['image'], video[:, :, 2:, :]))
            self.assertTrue(out['final'])

            with self.assertRaises(v2v.OutOfInputs):
                loader()

    def test_writer_handling(self):
        '''
        Tests basic handling of the :class:`Writer`.
        '''
        with tempfile.TemporaryDirectory() as temp_root:
            writer = v2v.Writer(temp_root, extension='png')

            image = np.ones((16, 16, 3), dtype=np.float32)
            writer({'image': image, 'names': ['00000']})
            temp_path_1 = os.path.join(temp_root, '00000.png')
            self.assertTrue(os.path.exists(temp_path_1))
            image = v2v.load(temp_path_1)
            self.assertEqual(image.shape, (16, 16, 3))
            self.assertTrue((np.abs(image - 1) < 1e-2).all())

            video = np.ones((16, 16, 2, 3), dtype=np.float32)
            writer({'image': video, 'names': ['00001', '00002']})
            temp_path_2 = os.path.join(temp_root, '00001.png')
            self.assertTrue(os.path.exists(temp_path_2), os.listdir(temp_root))
            temp_path_3 = os.path.join(temp_root, '00002.png')
            self.assertTrue(os.path.exists(temp_path_3), os.listdir(temp_root))

            with self.assertRaises(FileExistsError):
                writer({'image': image, 'names': ['00000']})

            # Test with suffix
            writer = v2v.Writer(temp_root, extension='png', suffix='_a')

            image = np.ones((16, 16, 3), dtype=np.float32)
            writer({'image': image, 'names': ['00003']})
            temp_path_3 = os.path.join(temp_root, '00003_a.png')
            self.assertTrue(os.path.exists(temp_path_3))
            image = v2v.load(temp_path_3)
            self.assertEqual(image.shape, (16, 16, 3))
            self.assertTrue((np.abs(image - 1) < 1e-2).all())

    def test_writer_buffer(self):
        images = np.stack([
            np.zeros((16, 16, 3), dtype=np.float32),
            np.full((16, 16, 3), 255. / 256., dtype=np.float32)
        ], axis=2)
        writer = v2v.Writer()

        with v2v.io._read_write_from_to_buffer():
            writer({'image': images, 'names': [None, None]})

        self.assertEqual(len(writer.buff), 2)
        for t in [0, 1]:
            self.assertEqual(writer.buff[t].shape, (16, 16, 3))
            self.assertEqual(writer.buff[t].dtype, np.float32)
            self.assertTrue((writer.buff[t] == images[:, :, t]).all())

    def test_writer_separate_bands(self):
        with tempfile.TemporaryDirectory() as temp_root:
            writer = v2v.Writer(
                temp_root, extension='png', separate_bands=True
            )

            image = np.ones((16, 16, 3), dtype=np.float32)
            writer({'image': image, 'names': ['00000']})

            video = np.ones((16, 16, 2, 3), dtype=np.float32)
            writer({'image': video, 'names': ['00001', '00002']})

            for i in range(3):
                for b in range(3):
                    temp_path = os.path.join(temp_root, f'0000{i}_{b}.png')
                    self.assertTrue(os.path.exists(temp_path))

            with self.assertRaises(FileExistsError):
                writer({'image': image, 'names': ['00000']})

    def test_writer_video_handling(self):
        '''
        Tests the :class:`Writer`'s ability to handle video.
        '''
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            writer = v2v.Writer(path, extension='mp4')
            writer({'image': video[:, :, :2, :], 'names': ['test', 'test']})
            writer({'image': video[:, :, 2:, :], 'names': ['test']})
            writer.release()

            rtn = v2v.load(path)

        self.assertEqual(rtn.dtype, np.float32)
        self.assertEqual(rtn.shape, (256, 256, 3, 3))
        self.assertTrue(_is_close(rtn, video))

        video = np.zeros((16, 16, 2, 3), dtype=np.float32)
        video[:, : 0, :] = (-4, 128, 260)

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            writer = v2v.Writer(path, extension='mp4')
            writer({'image': video, 'names': ['test', 'test']})
            writer.release()

            rtn = v2v.load(path)

        self.assertEqual(rtn.dtype, np.float32)
        self.assertEqual(rtn.shape, (16, 16, 2, 3))
        should_be = np.zeros((16, 16, 2, 3), dtype=np.float32)
        should_be[:, : 0, :] = (0, 128, 255)
        self.assertTrue(_is_close(rtn, should_be))

    def test_writer_video_separate_bands(self):
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.mp4')
            writer = v2v.Writer(path, extension='mp4', separate_bands=True)
            writer({'image': video[:, :, :2, :], 'names': ['test', 'test']})
            writer({'image': video[:, :, 2:, :], 'names': ['test']})
            writer.release()

            for b in range(3):
                path = os.path.join(temp_root, f'test_{b}.mp4')
                self.assertTrue(os.path.exists(path))

    def test_get_frame(self):
        '''
        Tests the :meth:`get_frame` method.
        '''
        video = _get_video()

        with tempfile.TemporaryDirectory() as temp_root:
            v2v.save(video, os.path.join(temp_root, 'test_0.mp4'))
            v2v.save(video[:, :, 0], os.path.join(temp_root, 'test_1.png'))
            v2v.save(video[:, :, 1], os.path.join(temp_root, 'test_2.png'))

            loader = v2v.Loader(temp_root, (256, 256), batch_size=2)

            self.assertTrue(len(loader) == 5, len(loader))

            for t in range(5):
                frame = loader.get_frame(t)
                self.assertTrue(_is_close(frame, video[:, :, t % 3]), t)
                frame = loader.get_frame(t, noscale=True)
                self.assertTrue(
                    _is_close(frame, (256 * video[:, :, t % 3]))
                )

            with self.assertRaises(ValueError):
                loader.get_frame(5)

        with tempfile.TemporaryDirectory() as temp_root:
            loader = v2v.Loader(temp_root, (256, 256), batch_size=2)
            with self.assertRaises(FileNotFoundError):
                loader.get_frame(0)

    def test_writer_path_inference(self):
        image = {
            'image': np.ones((32, 32, 2, 3), dtype=np.float32),
            'names': ['a', 'b'],
        }

        # Test directory out path.

        with tempfile.TemporaryDirectory() as temp_path:
            # No extension
            writer = v2v.Writer(temp_path)
            writer(image)
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'a.tif')))
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'b.tif')))

        with tempfile.TemporaryDirectory() as temp_path:
            # Specified extension - PNG
            writer = v2v.Writer(temp_path, extension='png')
            writer(image)
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'a.png')))
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'b.png')))

        with tempfile.TemporaryDirectory() as temp_path:
            # Specified extension - MP4
            writer = v2v.Writer(temp_path, extension='mp4')
            writer(image)
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'a.mp4')))
            self.assertFalse(os.path.exists(os.path.join(temp_path, 'b.mp4')))

        with tempfile.TemporaryDirectory() as temp_path:
            # Specified extension - MP4
            temp_path = os.path.join(temp_path, 'c.png')
            writer = v2v.Writer(temp_path, extension='png')
            with self.assertRaises(FileExistsError):
                writer(image)

        with tempfile.TemporaryDirectory() as temp_path:
            # Specified extension - MP4
            out_path = os.path.join(temp_path, 'c.mp4')
            writer = v2v.Writer(out_path, extension='mp4')
            writer(image)
            self.assertFalse(os.path.exists(os.path.join(temp_path, 'a.mp4')))
            self.assertFalse(os.path.exists(os.path.join(temp_path, 'b.mp4')))
            self.assertTrue(os.path.exists(os.path.join(temp_path, 'c.mp4')))

    def test_convert_and_scale_uint8(self):
        image = np.ones((2, 2, 3), dtype=np.uint8)
        out = v2v.io._convert_and_scale_uint8(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out == 1 / 256.).all())

        buff = np.zeros((2, 2, 3), dtype=np.float32)
        out = v2v.io._convert_and_scale_uint8(image, out=buff)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out == 1 / 256.).all())
        self.assertTrue((buff == 1 / 256.).all())

        image = np.ones((2, 2, 3), dtype=np.float64)
        out = v2v.io._convert_and_scale_uint8(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out == 1 / 256.).all())

        buff = np.zeros((2, 2, 3), dtype=np.float32)
        out = v2v.io._convert_and_scale_uint8(image, out=buff)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out == 1 / 256.).all())
        self.assertTrue((buff == 1 / 256.).all())

    def test_error_handling(self):
        with tempfile.TemporaryDirectory() as temp_root:
            loader = v2v.Loader(temp_root, (256, 256), batch_size=1)
            with self.assertRaises(FileNotFoundError):
                loader()


if __name__ == '__main__':
    unittest.main()
