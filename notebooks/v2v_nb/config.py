from copy import copy
from glob import glob
import json
import os
from tkinter import Label
from typing import Any, Dict, Optional, Tuple
import warnings

import cv2
import ipywidgets as widgets
import numpy as np
from PIL import Image

import video2vision as v2v

__all__ = ['clear_all', 'get_config', 'Config', 'ParamNotSet', 'save_defaults']


MAX_TEXT_LENGTH = 40

DEFAULTS_PATH = 'defaults.json'


# TODO: Add defaults

PARAM_TYPES: Dict[str, str] = {
    'experiment_name': 'string',
    'align_pipe_path': 'path',
    'vis_path': 'path',
    'uv_path': 'path',
    'uv_aligned_path': 'path',
    'animal_out_path': 'path',
    'human_out_path': 'path',
    'vis_linearization_path': 'path',
    'uv_linearization_path': 'path',
    'linearization_values_path': 'path',
    'camera_path': 'path',
    'is_sony_camera': 'bool',
    'vis_test_path': 'path',
    'uv_test_path': 'path',
    'test_values_path': 'path',
    'linearization_auto_op_path': 'path',
    'test_auto_op_path': 'path',
    'animal_sensitivity_path': 'path',
    'sense_converter_path': 'path',
    'shift': 'int',
    'coe': 'array',
    'batch_size': 'int',
    # This is used in the alignment-pipeline-builder notebook
    'save_align_pipe_path': 'path',
    'build_video_pipeline': 'bool',
    # These are used in testing, not in operations
    'test_array': 'array',
    'test_bool': 'bool',
    'test_path': 'path',
    'test_str': 'str',
    'test_int': 'int',
}

# Captions for the labels in the display window.
PARAM_CAPTIONS: Dict[str, str] = {
    'experiment_name': 'Experiment Name',
    'align_pipe_path': 'Alignment Pipeline',
    'vis_path': 'Visible Input Path',
    'uv_path': 'UV Input Path',
    'uv_aligned_path': 'Aligned UV Output Path',
    'animal_out_path': 'Animal Vision Output Path',
    'human_out_path': 'Human Vision Output Path',
    'vis_linearization_path': 'Visible Linearization Path',
    'uv_linearization_path': 'UV Linearization Path',
    'linearization_values_path': 'Linearization Sample Values Path',
    'camera_path': 'Camera Sensitivities Path',
    'is_sony_camera': 'Use Sony SLog3 Linearization?',
    'vis_test_path': 'Visible Test Path',
    'uv_test_path': 'UV Test Path',
    'test_values_path': 'Test Sample Values Path',
    'linearization_auto_op_path': 'Linearization Sample Autolocator Path',
    'test_auto_op_path': 'Test Sample Autolocator Path',
    'animal_sensitivity_path': 'Animal Sensitivities Path',
    'sense_converter_path': 'Sense Converter Path',
    'shift': 'Time Shift',
    'coe': 'Alignment Warp',
    'linearization': 'Linearization',
    'sense_converter': 'Sense Converter',
    # This is used in the alignment-pipeline-builder notebook
    'save_align_pipe_path': 'Alignment Pipeline',
    'build_video_pipeline': 'Build Video Pipeline?',
    # These are used in testing, not in operations
    'test_array': 'You should never see this',
    'test_bool': 'You should never see this',
    'test_path': 'You should never see this',
    'test_str': 'You should never see this',
    'test_int': 'You should never see this',
}

PARAM_PARSE = {'coe': np.array}


class Config:
    '''
    This will be a unique global object that both holds the current
    configuration options. It should not be accessed directly; instead, it
    should be reached through :func:`get_params_window` below.
    '''
    # Actual values of the configuration options. None means not set yet.
    _values: Dict[str, Any] = {k: None for k in PARAM_TYPES}
    # This will contain pointers to the Label objects in the display window
    # once they are created.
    _popup_labels: Dict[str, Optional[Label]] = copy(_values)
    # This will contain pointers to the Label objects in the display window
    # once they are created.
    _nb_labels: Dict[str, Optional[widgets.Text]] = copy(_values)

    def __init__(self):
        self._image_size = None
        self._out_extension = None
        self._cache_ready = False
        self._values['batch_size'] = 16

        if os.path.exists(DEFAULTS_PATH):
            try:
                with open(DEFAULTS_PATH, 'r') as cache_file:
                    cache = json.load(cache_file)

                for k, v in cache.items():
                    if k in PARAM_PARSE:
                        v = PARAM_PARSE[k](v)
                    self[k] = v

            except json.JSONDecodeError:
                warnings.warning(
                    f'Malformed defaults file: {DEFAULTS_PATH}. Recommend '
                    f'deleting this file.'
                )

        self._cache_ready = True

    def __contains__(self, k: str) -> bool:
        return k in self._values

    def __getitem__(self, k: str):
        return self._values[k]

    def __setitem__(self, k: str, v: Any):
        if k in self._values:
            self._values[k] = v

            if (k == 'experiment_name') and self._cache_ready:
                self.load_cache()
            if self._cache_ready:
                self.save_cache()

            text = self._label_text(k)
            if self._popup_labels[k] is not None:
                self._popup_labels[k].config(
                    text=f'{self._label_captions[k]}: {text}'
                )
            if self._nb_labels[k] is not None:
                # TODO: Move this logic into the widget
                if PARAM_TYPES[k] == 'bool':
                    v = bool(v)
                elif PARAM_TYPES[k] != 'array':
                    v = '' if (v is None) else str(v)
                self._nb_labels[k].value = v
        else:
            raise KeyError(k)

    def __delitem__(self, k: str):
        if k in self._values:
            raise RuntimeError(f'Cannot delete {k}')
        else:
            raise KeyError(k)

    @property
    def _cache_path(self) -> str:
        return os.path.join(self['experiment_name'], 'cache.json')

    @property
    def has_alignment(self) -> bool:
        return (self['coe'] is not None) and (self['shift'] is not None)

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        if self._image_size is None:
            for k in [
                'vis_path', 'uv_path', 'uv_aligned_path', 'human_out_path',
                'animal_out_path',
            ]:
                if self[k] is not None:
                    path = self[k]
                    if os.path.isdir(path):
                        path = os.path.join(path, '*')
                    try:
                        path = glob(path)[0]
                    except IndexError:
                        continue
                    else:
                        self._image_size = _get_size(path)
                        break
            else:
                if self['align_pipe_path'] is None:
                    raise ParamNotSet('align_pipe_path')
                align_pipe = v2v.load_pipeline(self['align_pipe_path'])
                self._image_size = align_pipe.get_loaders()[0].expected_size
                self._out_extension = align_pipe.get_writers()[0].extension

        return tuple(self._image_size)

    @property
    def is_3band_out(self) -> bool:
        if self['animal_sensitivity_path'] is None:
            raise ParamNotSet('animal_sensitivity_path')
        animal_sense = np.genfromtxt(
            self['animal_sensitivity_path'], skip_header=True, delimiter=','
        )
        n_bands = animal_sense.shape[1] - 1
        return (n_bands == 3)

    def _label_text(self, k: str) -> str:
        text = '' if (self._values[k] is None) else str(self._values[k])
        if len(text) > MAX_TEXT_LENGTH:
            text = f'{text[:MAX_TEXT_LENGTH - 4]} ...'
        return text

    @property
    def linearization_crosshair_cache_path(self) -> Optional[str]:
        if self.use_cache:
            return os.path.join(self['experiment_name'], 'line_cross.json')
        else:
            return None

    def load_cache(self):
        if self.use_cache and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r') as cache_file:
                    cache = json.load(cache_file)

                for k in self._values.keys():
                    if k == 'experiment_name':
                        continue
                    v = cache.get(k, None)
                    if (k in PARAM_PARSE) and (v is not None):
                        v = PARAM_PARSE[k](v)
                    self[k] = v

            except json.JSONDecodeError:
                raise RuntimeError(
                    f'Malformed cache file: please delete {self._cache_path}'
                )

    @property
    def out_extension(self) -> Optional[str]:
        if self._out_extension is None:
            if self['align_pipe_path'] is None:
                raise ParamNotSet('align_pipe_path')
            align_pipe = v2v.load_pipeline(self['align_pipe_path'])
            self._image_size = align_pipe.get_loaders()[0].expected_size
            self._out_extension = align_pipe.get_writers()[0].extension

        return self._out_extension

    @property
    def out_path(self) -> Optional[str]:
        for k in ('uv_path', 'vis_path', 'uv_aligned_path', 'align_pipe_path'):
            if self[k] is None:
                raise ParamNotSet(k)

        if self['uv_aligned_path'].endswith(f'.{self.out_extension}'):
            return self['uv_aligned_path']
        else:
            uv_name, _ = os.path.splitext(os.path.basename(self['uv_path']))
            vis_name, _ = os.path.splitext(os.path.basename(self['vis_path']))
            out_name = f'{uv_name}-{vis_name}.{self.out_extension}'
            return os.path.join(self['uv_aligned_path'], out_name)

    def save_cache(self):
        if self.use_cache:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)

            out = {}
            for k, v in self._values.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, v2v.Operator):
                    v = v._to_json()
                out[k] = v

            with open(self._cache_path, 'w') as cache_file:
                json.dump(out, cache_file)

    @property
    def test_crosshair_cache_path(self) -> Optional[str]:
        if self.use_cache:
            return os.path.join(self['experiment_name'], 'test_cross.json')
        else:
            return None

    @property
    def use_cache(self) -> bool:
        return bool(self['experiment_name'])


class ParamNotSet(Exception):
    pass


# We use global variables to store the config object to avoid accidentally
# creating multiple of them.
_config = Config()


def get_config(clear: bool = False) -> Config:
    global _config
    if clear:
        clear_all()
    return _config


def save_defaults():
    global _config

    out = {}
    for k, v in _config._values.items():
        if (k == 'experiment_name') or (v is None):
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, v2v.Operator):
            v = v._to_json()
        out[k] = v

    with open(DEFAULTS_PATH, 'w') as cache_file:
        json.dump(out, cache_file)


def clear_all():
    global _config

    # Do this first to avoid overwriting the cache
    _config['experiment_name'] = None

    for k in PARAM_TYPES.keys():
        _config[k] = None

    _config['batch_size'] = 16
    _config._out_extension = _config._image_size = None


def _get_size(path: str) -> Tuple[int, int]:
    '''
    Extracts and returns the size of an image or video on disk, as (height,
    width).
    '''
    if path.lower().endswith('.mp4'):
        reader = cv2.VideoCapture(path)
        w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
    else:
        with Image.open(path) as image:
            return image.size
