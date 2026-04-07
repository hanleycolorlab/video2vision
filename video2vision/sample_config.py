'''
Utilities for managing sample configuration files and discovering video pairs
in the batch processing workflow.

Sample directories follow this structure::

    videos/samples/
        001/
            config.json
            VIS_001.MP4
            UV_001.MP4
            calibration/
                VIS_001.MP4
                UV_001.MP4
        002/
            ...
'''

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


__all__ = [
    'find_sample_dirs', 'find_video_pair', 'load_sample_config',
    'save_sample_config',
]

# Default location for sample directories
DEFAULT_SAMPLES_DIR = 'videos/samples'


def load_sample_config(
    sample_id: str, samples_dir: str = DEFAULT_SAMPLES_DIR
) -> Optional[Dict]:
    '''
    Load configuration for a sample from its config.json file.

    Args:
        sample_id (str): The sample identifier (e.g. '001').
        samples_dir (str): Path to the directory containing sample
        subdirectories.

    Returns:
        dict or None: The parsed configuration, or None if config.json
        does not exist.
    '''
    config_path = Path(samples_dir) / sample_id / 'config.json'
    if not config_path.exists():
        return None
    with open(config_path, 'r') as f:
        return json.load(f)


def save_sample_config(
    sample_id: str, config: Dict, samples_dir: str = DEFAULT_SAMPLES_DIR
):
    '''
    Save configuration for a sample to its config.json file.

    Args:
        sample_id (str): The sample identifier (e.g. '001').
        config (dict): The configuration to save.
        samples_dir (str): Path to the directory containing sample
        subdirectories.
    '''
    config_path = Path(samples_dir) / sample_id / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def find_video_pair(
    sample_dir: str, use_calibration: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    '''
    Find a VIS/UV video pair in a sample directory.

    Searches for files matching ``VIS_*.MP4`` and ``UV_*.MP4`` (case
    insensitive extension). Returns the first match of each.

    Args:
        sample_dir (str): Path to the sample directory.
        use_calibration (bool): If True, look in the ``calibration/``
        subdirectory instead.

    Returns:
        tuple: (vis_path, uv_path) as strings, or (None, None) if not
        found.
    '''
    sample_dir = Path(sample_dir)
    if use_calibration:
        sample_dir = sample_dir / 'calibration'

    if not sample_dir.exists():
        return None, None

    vis_videos = (
        sorted(sample_dir.glob('VIS_*.MP4'))
        + sorted(sample_dir.glob('VIS_*.mp4'))
    )
    uv_videos = (
        sorted(sample_dir.glob('UV_*.MP4'))
        + sorted(sample_dir.glob('UV_*.mp4'))
    )

    if not vis_videos or not uv_videos:
        return None, None

    return str(vis_videos[0]), str(uv_videos[0])


def find_sample_dirs(
    samples_dir: str = DEFAULT_SAMPLES_DIR,
    sample_ids: Optional[List[str]] = None,
) -> List[Tuple[str, Path]]:
    '''
    Find sample directories, optionally filtered to specific IDs.

    Args:
        samples_dir (str): Path to the directory containing sample
        subdirectories.
        sample_ids (optional, list of str): If provided, only return
        directories matching these IDs. If None, returns all
        subdirectories sorted by name.

    Returns:
        list of (sample_id, sample_path) tuples.
    '''
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return []

    if sample_ids is not None:
        dirs = []
        for sid in sample_ids:
            d = samples_path / sid
            if d.is_dir():
                dirs.append((sid, d))
        return dirs

    return sorted(
        [(d.name, d) for d in samples_path.iterdir() if d.is_dir()],
        key=lambda x: x[0],
    )
