import json
import os
import tempfile
import unittest

from video2vision.sample_config import (
    find_sample_dirs, find_video_pair, load_sample_config,
    save_sample_config,
)


class SampleConfigTest(unittest.TestCase):
    def test_load_missing_config(self):
        '''Returns None when config.json does not exist.'''
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '001'))
            result = load_sample_config('001', tmp)
            self.assertIsNone(result)

    def test_load_and_save_config(self):
        '''Round-trips a config through save then load.'''
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '001'))
            config = {
                'sample_id': '001',
                'flip_main': 'horizontal',
            }
            save_sample_config('001', config, tmp)

            # Verify file was created
            config_path = os.path.join(tmp, '001', 'config.json')
            self.assertTrue(os.path.exists(config_path))

            # Verify contents
            loaded = load_sample_config('001', tmp)
            self.assertEqual(loaded, config)

    def test_save_overwrites_existing(self):
        '''Saving overwrites an existing config.'''
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '001'))
            save_sample_config('001', {'a': 1}, tmp)
            save_sample_config('001', {'b': 2}, tmp)
            loaded = load_sample_config('001', tmp)
            self.assertEqual(loaded, {'b': 2})

    def test_save_json_formatting(self):
        '''Saved config uses indent=2 for readability.'''
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '001'))
            save_sample_config('001', {'key': 'val'}, tmp)
            path = os.path.join(tmp, '001', 'config.json')
            with open(path) as f:
                raw = f.read()
            # indent=2 produces multi-line output
            self.assertIn('\n', raw)
            self.assertEqual(json.loads(raw), {'key': 'val'})


class FindVideoPairTest(unittest.TestCase):
    def _create_video_files(self, directory, prefix=''):
        '''Helper to create dummy video files.'''
        for name in [f'{prefix}VIS_001.MP4', f'{prefix}UV_001.MP4']:
            path = os.path.join(directory, name)
            with open(path, 'w') as f:
                f.write('dummy')

    def test_find_pair_in_main_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = os.path.join(tmp, '001')
            os.makedirs(sample_dir)
            self._create_video_files(sample_dir)

            vis, uv = find_video_pair(sample_dir)
            self.assertIsNotNone(vis)
            self.assertIsNotNone(uv)
            self.assertIn('VIS_', vis)
            self.assertIn('UV_', uv)

    def test_find_pair_in_calibration_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = os.path.join(tmp, '001')
            cal_dir = os.path.join(sample_dir, 'calibration')
            os.makedirs(cal_dir)
            self._create_video_files(cal_dir)

            vis, uv = find_video_pair(sample_dir, use_calibration=True)
            self.assertIsNotNone(vis)
            self.assertIsNotNone(uv)
            self.assertIn('calibration', vis)

    def test_returns_none_for_missing_dir(self):
        vis, uv = find_video_pair('/nonexistent/path')
        self.assertIsNone(vis)
        self.assertIsNone(uv)

    def test_returns_none_when_no_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            vis, uv = find_video_pair(tmp)
            self.assertIsNone(vis)
            self.assertIsNone(uv)

    def test_returns_none_when_only_vis(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'VIS_001.MP4'), 'w') as f:
                f.write('dummy')
            vis, uv = find_video_pair(tmp)
            self.assertIsNone(vis)
            self.assertIsNone(uv)

    def test_calibration_returns_none_when_no_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            vis, uv = find_video_pair(tmp, use_calibration=True)
            self.assertIsNone(vis)
            self.assertIsNone(uv)

    def test_lowercase_extension(self):
        '''Finds videos with lowercase .mp4 extension.'''
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['VIS_001.mp4', 'UV_001.mp4']:
                with open(os.path.join(tmp, name), 'w') as f:
                    f.write('dummy')
            vis, uv = find_video_pair(tmp)
            self.assertIsNotNone(vis)
            self.assertIsNotNone(uv)


class FindSampleDirsTest(unittest.TestCase):
    def test_find_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['001', '002', '003']:
                os.makedirs(os.path.join(tmp, name))
            # Also create a file (should be ignored)
            with open(os.path.join(tmp, 'notes.txt'), 'w') as f:
                f.write('dummy')

            dirs = find_sample_dirs(tmp)
            self.assertEqual(len(dirs), 3)
            ids = [d[0] for d in dirs]
            self.assertEqual(ids, ['001', '002', '003'])

    def test_filter_by_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['001', '002', '003']:
                os.makedirs(os.path.join(tmp, name))

            dirs = find_sample_dirs(tmp, sample_ids=['001', '003'])
            self.assertEqual(len(dirs), 2)
            ids = [d[0] for d in dirs]
            self.assertEqual(ids, ['001', '003'])

    def test_missing_id_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '001'))
            dirs = find_sample_dirs(tmp, sample_ids=['001', '999'])
            self.assertEqual(len(dirs), 1)

    def test_nonexistent_dir_returns_empty(self):
        dirs = find_sample_dirs('/nonexistent/path')
        self.assertEqual(dirs, [])


if __name__ == '__main__':
    unittest.main()
