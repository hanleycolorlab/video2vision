import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tabulate import tabulate

import video2vision as v2v

from .config import get_config, PARAM_CAPTIONS, ParamNotSet
from .displays import DisplayBox, GhostBox, SelectorBox
from .utils import (
    coefficient_of_determination,
    get_cache_path,
    get_loader,
    get_shift,
    load_csv,
    load_operator,
    make_displayable,
    mean_absolute_error,
    signal_to_noise_ratio
)


__all__ = [
    'build_and_run_alignment_pipeline', 'build_and_save_alignment_pipeline',
    'build_and_run_full_pipeline', 'build_linearizer', 'build_coarse_warp',
    'evaluate_conversion', 'evaluate_samples',
    'make_example_linearization_images', 'make_final_displaybox',
    'make_ghostbox', 'make_initial_displaybox', 'make_selectorbox',
]


def build_and_run_alignment_pipeline():
    '''
    Builds and runs a :class:`video2vision.Pipeline` to perform alignment
    between the visible and UV imagery.
    '''
    config = get_config()

    for k in ['align_pipe_path', 'uv_path', 'vis_path', 'uv_aligned_path']:
        if not config[k]:
            print(f'Please specify {PARAM_CAPTIONS[k].lower()}')
            return

    # Build alignment pipe
    align_pipe = v2v.load_pipeline(config['align_pipe_path'])
    config._image_size = align_pipe.get_loaders()[0].expected_size
    config._out_extension = align_pipe.get_writers()[0].extension
    align_pipe.set_all_paths(
        config['uv_path'], config['vis_path'], config['uv_aligned_path'],
    )
    align_pipe.set_batch_size(config['batch_size'])

    align_idx, = (
        i for i in align_pipe.nodes
        if isinstance(align_pipe.nodes[i]['operator'], v2v.AutoAlign)
    )
    align_op = align_pipe.nodes[align_idx]['operator']

    # If cached values are available, use them
    if (config['coe'] is not None) and (config['shift'] is None):
        print(
            'Found spatial warp but not temporal alignment, which is not '
            'supported. Please either clear the warp so it can be refitted, or'
            ' manually set the temporal shift.'
        )
        return
    align_op.coe = config['coe']
    align_op.output_size = config.image_size
    if isinstance(align_op, v2v.AutoTemporalAlign):
        align_op.time_shift = config['shift']

    # Check if output already exists
    if os.path.exists(config.out_path):
        if (config['coe'] is None) or (config['shift'] is None):
            print(
                f'Output file {config.out_path} already exists but alignment '
                f'parameters are not in config. Please delete the output file '
                f'so that alignment pipeline can be rerun.'
            )
            return

        print('Pipeline already ran')

    else:
        try:
            align_pipe.run()
        except v2v.AlignmentNotFound:
            print('Failed to find alignment. Check inputs.')
            return
        else:
            print('Pipeline complete')

    config['coe'] = align_op.coe
    if isinstance(align_op, v2v.AutoTemporalAlign):
        config['shift'] = align_op.time_shift
    else:
        config['shift'] = 0


def build_and_save_alignment_pipeline(warp_op: v2v.Warp):
    '''
    Builds an alignment pipeline from a coarse :class:`video2vision.Warp`
    operator and saves to disk.
    '''
    config = get_config()

    if warp_op is None:
        print('Warp operator must be built first.')
        return
    if config['save_align_pipe_path'] is None:
        print('Please specify path to write alignment pipeline to.')
        return
    if os.path.exists(config['save_align_pipe_path']):
        print(
            'The alignment pipeline already exists. Cowardly refusing to '
            'overwrite it.'
        )
        return
    if config['vis_path'] is None:
        print('Please specify path to visible image.')
        return

    if config['build_video_pipeline']:
        align_op = v2v.AutoTemporalAlign(
            time_shift_range=[-10, 10], bands=[[0, 1, 2], []]
        )
        write_op = v2v.Writer(extension='mp4')
    else:
        align_op = v2v.AutoAlign(num_votes=4, bands=[[0, 1, 2], []])
        write_op = v2v.Writer(extension='png')

    pipe = v2v.Pipeline()
    uv_loader_idx = pipe.add_operator(v2v.Loader(None, config.image_size))
    vis_loader_idx = pipe.add_operator(v2v.Loader(None, config.image_size))
    coarse_align_idx = pipe.add_operator(warp_op)
    fine_align_idx = pipe.add_operator(align_op)
    write_idx = pipe.add_operator(write_op)

    pipe.add_edge(uv_loader_idx, coarse_align_idx, in_slot=0)
    pipe.add_edge(coarse_align_idx, fine_align_idx, in_slot=0)
    pipe.add_edge(vis_loader_idx, fine_align_idx, in_slot=1)
    pipe.add_edge(fine_align_idx, write_idx, in_slot=0)
    pipe.save(config['save_align_pipe_path'])

    print('Done! You can close the notebook.')


def build_and_run_full_pipeline(line_op: v2v.ElementwiseOperator):
    config = get_config()

    if (config['coe'] is None) or (config['shift'] is None):
        print('Alignment must be run before linearization.')
        return
    if line_op is None:
        line_op_cache_path = os.path.join(
            config['experiment_name'] or '', 'line_op.json'
        )
        if config.use_cache and os.path.exists(line_op_cache_path):
            print('Reloading linearizer from cache.')
            line_op = load_operator(line_op_cache_path)
        else:
            print('Linearization operator must be built before linearization')
            return
    for k in [
        'align_pipe_path', 'uv_path', 'vis_path', 'animal_out_path',
        'sense_converter_path', 'human_out_path', 'animal_sensitivity_path',
    ]:
        if not config[k]:
            print(f'Please specify {PARAM_CAPTIONS[k].lower()}')
            return

    full_pipe = v2v.load_pipeline(config['align_pipe_path'])

    align_idx, = (
        i for i in full_pipe.nodes
        if isinstance(full_pipe.nodes[i]['operator'], v2v.AutoAlign)
    )
    align_op = full_pipe.nodes[align_idx]['operator']
    align_op.coe = config['coe']
    align_op.output_size = config.image_size
    if isinstance(align_op, v2v.AutoTemporalAlign):
        align_op.shift = config['shift']
    align_op._concatenate.bands = [[2], [0, 1, 2]]

    line_idx = full_pipe.add_operator(line_op)
    write_idx = next(
        i for i in full_pipe.nodes
        if isinstance(full_pipe.nodes[i]['operator'], v2v.Writer)
    )
    write_op = full_pipe.nodes[write_idx]['operator']
    write_op.separate_bands = not config.is_3band_out
    full_pipe.remove_edge(align_idx, write_idx)
    full_pipe.add_edge(align_idx, line_idx, in_slot=0)

    sense_converter = load_operator(config['sense_converter_path'])
    sense_idx = full_pipe.add_operator(sense_converter)
    full_pipe.add_edge(line_idx, sense_idx, in_slot=0)
    full_pipe.add_edge(sense_idx, write_idx, in_slot=0)

    write_op_2 = v2v.Writer(extension=write_op.extension)
    write_idx_2 = full_pipe.add_operator(write_op_2)
    sel_idx = full_pipe.add_operator(v2v.ConcatenateOnBands([[1, 2, 3]]))
    full_pipe.add_edge(line_idx, sel_idx, in_slot=0)
    full_pipe.add_edge(sel_idx, write_idx_2, in_slot=0)

    full_pipe.set_batch_size(config['batch_size'] // 2)
    full_pipe.set_all_paths(
        config['uv_path'], config['vis_path'], config['animal_out_path'],
        config['human_out_path'],
    )

    full_pipe.run()
    print('Pipeline complete')


def build_coarse_warp(vis_selector: SelectorBox, uv_selector: SelectorBox):
    '''
    This function is used in the alignment pipeline builder notebook. Given a
    pair of :class:`SelectorBox`, we construct the coarse warp operation and
    generate an example coarsely-aligned image, returning both.
    '''
    config = get_config()

    if (vis_selector is None) or (uv_selector is None):
        print(
            'Please run both selectors and choose tie points before building '
            'the warp.'
        )
        return
    if config['vis_path'] is None:
        print('Please specify path to visible image.')
        return

    n_uv_crosshairs = len(uv_selector.crosshairs)
    n_vis_crosshairs = len(vis_selector.crosshairs)

    if n_uv_crosshairs != n_vis_crosshairs:
        print('You need to select the same number of samples in each image.')
        return
    if n_uv_crosshairs < 4:
        print('At least four tie points must be selected in each image.')
        return

    warp_op = v2v.Warp.build_from_tiepoints(
        uv_selector.crosshairs,
        vis_selector.crosshairs,
        config.image_size,
    )

    # The _original_image is in uint8, so will need to be rescaled prior to and
    # after the warp.
    uv_image = uv_selector._original_image.astype(np.float32) / 256.
    warped_uv_image = warp_op({'image': uv_image})['image']
    warped_uv_image = np.clip(256 * warped_uv_image, 0, 255).astype(np.uint8)
    # Now resize down to match vis_image
    if (uv_selector.h, uv_selector.w) != warped_uv_image.shape[:2]:
        warped_uv_image = cv2.resize(
            warped_uv_image, (uv_selector.w, uv_selector.h)
        )

    display_image = np.concatenate(
        (vis_selector._cached_image, warped_uv_image), axis=1
    )

    return warp_op, Image.fromarray(display_image)


def build_linearizer(vis_selector: SelectorBox, uv_selector: SelectorBox) \
        -> v2v.ElementwiseOperator:
    '''
    Builds and returns a linearizer operator.
    '''
    config = get_config()

    cache_path = os.path.join(config['experiment_name'] or '', 'line_op.json')

    if config.use_cache and os.path.exists(cache_path):
        print('Reloading from cache.')
        return load_operator(cache_path)
    if (vis_selector is None) or (uv_selector is None):
        print('Please select samples before building linearizer.')
        return
    for k in ['linearization_values_path', 'camera_path', 'is_sony_camera']:
        if config[k] is None:
            print(f'Please specify {PARAM_CAPTIONS[k].lower()}')
            return

    sample_ref = load_csv(config['linearization_values_path'])
    camera_sense = load_csv(config['camera_path'])
    expected_values = sample_ref.T.dot(camera_sense)

    vis_samples, vis_drop = vis_selector.get_samples()
    uv_samples, uv_drop = uv_selector.get_samples()

    if (vis_drop != uv_drop).any():
        print('Selection mismatch between visual and UV; please correct')
        return

    samples = np.concatenate((uv_samples[:, 2:], vis_samples), axis=1)
    samples = samples[~vis_drop]

    expected_values = sample_ref.T.dot(camera_sense)

    if config['is_sony_camera']:
        line_op = v2v.PowerLaw([
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
            [0.0047058172145495476, 4185.031519941784, -0.01,
             0.16736099187966763],
        ])
        linearized_sample_values = line_op.apply_values(samples)
        for band in range(4):
            scale = float(np.linalg.lstsq(
                linearized_sample_values[:, [band]],
                expected_values[:, [band]],
                rcond=None
            )[0])
            line_op.funcs[band].scale *= scale
            line_op.funcs[band].shift *= scale

    else:
        line_op = v2v.build_linearizer(
            samples, expected_values, method='power'
        )

    if config.use_cache:
        with open(cache_path, 'w') as cache_file:
            json.dump(line_op._to_json(), cache_file)

    return line_op


def evaluate_conversion(line_op: v2v.ElementwiseOperator,
                        values_path: str, vis_selector: SelectorBox,
                        uv_selector: SelectorBox) \
        -> Tuple[np.ndarray, np.ndarray, str]:
    config = get_config()

    if line_op is None:
        print('Linearizer must be built before evaluating.')
        return
    if values_path is None:
        print('Please specify path to sample values.')
        return
    for k in ['sense_converter_path', 'animal_sensitivity_path']:
        if not config[k]:
            print(f"Please specify {PARAM_CAPTIONS[k].lower()}")
            return
    if (vis_selector is None) or (uv_selector is None):
        print('Please select samples before performing evaluation.')
        return

    sample_ref = load_csv(values_path)
    animal_sense = load_csv(config['animal_sensitivity_path'])
    expected_values = sample_ref.T.dot(animal_sense)

    vis_samples, vis_drop = vis_selector.get_samples()
    uv_samples, uv_drop = uv_selector.get_samples()

    if (vis_drop != uv_drop).any():
        print('Selection mismatch between visual and UV; please correct')
        return

    samples = np.concatenate((uv_samples[:, 2:], vis_samples), axis=1)

    samples = samples[~vis_drop]
    sample_ref = sample_ref[:, ~vis_drop]
    expected_values = expected_values[~vis_drop]

    linearized_values = line_op.apply_values(samples)
    sense_converter = load_operator(config['sense_converter_path'])
    converted_values = linearized_values.dot(sense_converter.mat)
    n_bands = animal_sense.shape[1]

    table = list(zip(
        [f'Band {band}' for band in range(n_bands)],
        [mean_absolute_error(converted_values[:, band],
                             expected_values[:, band])
         for band in range(n_bands)],
        [coefficient_of_determination(converted_values[:, band],
                                      expected_values[:, band])
         for band in range(n_bands)],
    ))
    table = tabulate(table, headers=['Color', 'MAE', 'R2'])

    return expected_values, converted_values, table


def evaluate_samples(line_op: v2v.ElementwiseOperator,
                     values_path: str, vis_selector: SelectorBox,
                     uv_selector: SelectorBox) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    config = get_config()

    if line_op is None:
        print('Linearizer must be built before evaluating.')
        return
    if values_path is None:
        print('Please specify path to sample values.')
        return
    if not config['camera_path']:
        print(f"Please specify {PARAM_CAPTIONS['camera_path'].lower()}")
        return
    if (vis_selector is None) or (uv_selector is None):
        print('Please select samples before performing evaluation.')
        return

    sample_ref = load_csv(values_path)
    camera_sense = load_csv(config['camera_path'])
    expected_values = sample_ref.T.dot(camera_sense)

    vis_samples, vis_drop = vis_selector.get_samples()
    uv_samples, uv_drop = uv_selector.get_samples()

    if (vis_drop != uv_drop).any():
        print('Selection mismatch between visual and UV; please correct')
        return

    samples = np.concatenate((uv_samples[:, 2:], vis_samples), axis=1)
    samples = samples[~vis_drop]

    sample_ref = sample_ref[:, ~vis_drop]
    expected_values = expected_values[~vis_drop]

    linearized_values = line_op.apply_values(samples)

    table = list(zip(
        ['UV', 'Blue', 'Green', 'Red'],
        [mean_absolute_error(linearized_values[:, band],
                             expected_values[:, band])
         for band in range(4)],
        [coefficient_of_determination(linearized_values[:, band],
                                      expected_values[:, band])
         for band in range(4)],
        [signal_to_noise_ratio(linearized_values[:, band],
                               expected_values[:, band])
         for band in range(4)]
    ))
    table = tabulate(table, headers=['Color', 'MAE', 'R2', 'SNR'])

    return expected_values, linearized_values, samples, table


def make_example_linearization_images(line_op: v2v.ElementwiseOperator) \
        -> Image:
    config = get_config()

    for k in ['align_pipe_path', 'uv_path', 'vis_path']:
        if config[k] is None:
            print(f'Please specify {PARAM_CAPTIONS[k].lower()}')
            return
    if (config['coe'] is None) or (config['shift'] is None):
        print('Alignment must be run before linearization.')
        return
    if line_op is None:
        if config.use_cache and os.path.exists(config.line_op_cache_path):
            print('Reloading linearizer from cache.')
            line_op = load_operator(config.line_op_cache_path)
        else:
            print('Linearization operator must be built before linearization')
            return

    line_pipe = v2v.load_pipeline(config['align_pipe_path'])
    align_idx, = (
        i for i in line_pipe.nodes
        if isinstance(line_pipe.nodes[i]['operator'], v2v.AutoAlign)
    )
    align_op = line_pipe.nodes[align_idx]['operator']
    align_op.coe = config['coe']
    align_op.output_size = config.image_size
    align_op._concatenate.bands = [[2], [0, 1, 2]]
    if isinstance(align_op, v2v.AutoTemporalAlign):
        align_op.time_shift = 0

    line_idx = line_pipe.add_operator(line_op)

    write_idx = next(
        i for i in line_pipe.nodes
        if isinstance(line_pipe.nodes[i]['operator'], v2v.Writer)
    )
    line_pipe.remove_edge(align_idx, write_idx)
    line_pipe.add_edge(align_idx, line_idx, in_slot=0)
    line_pipe.add_edge(line_idx, write_idx, in_slot=0)

    vis_image = get_loader('vis_path').get_frame(max(config['shift'], 0))
    uv_image = get_loader('uv_path').get_frame(max(-config['shift'], 0))
    image = line_pipe(uv_image, vis_image)

    return make_displayable(image[..., :1], image[..., 1:])


def make_final_displaybox() -> DisplayBox:
    config = get_config()

    try:
        if config.is_3band_out:
            loaders = [get_loader('animal_out_path')]

        else:
            loaders = [
                v2v.Loader(os.path.join(config['animal_out_path'], f'*_{b}.*'),
                           config.image_size, num_channels=1)
                for b in range(3)
            ]

        return DisplayBox(
            *loaders, get_loader('human_out_path'),
            output_size=0.25,
        )
    except FileNotFoundError as err:
        print(f'Could not find file {err.args[0]}; please check paths')
    except ParamNotSet as err:
        print(f'Please specify {PARAM_CAPTIONS[err.args[0]].lower()}.')


def make_ghostbox() -> GhostBox:
    config = get_config()

    if config['shift'] is None:
        print('Please run alignment first.')
        return

    try:
        return GhostBox(
            get_loader('vis_path'),
            get_loader('uv_aligned_path'),
            shifts=(max(-config['shift'], 0), 0),
            output_size=0.5,
        )
    except FileNotFoundError as err:
        print(f'Could not find file {err.args[0]}; please check paths')
    except ParamNotSet as err:
        print(f'Please specify {PARAM_CAPTIONS[err.args[0]].lower()}.')


def make_initial_displaybox() -> DisplayBox:
    config = get_config()

    if config['shift'] is None:
        print('Please run alignment first.')
        return

    try:
        return DisplayBox(
            get_loader('vis_path'),
            get_loader('uv_aligned_path'),
            get_loader('uv_path'),
            shifts=(max(-config['shift'], 0), 0, 0),
            output_size=0.15,
        )
    except FileNotFoundError as err:
        print(f'Could not find file {err.args[0]}; please check paths')
    except ParamNotSet as err:
        print(f'Please specify {PARAM_CAPTIONS[err.args[0]].lower()}.')


def make_selectorbox(which: str, copy_from: Optional[SelectorBox] = None,
                     require_alignment: bool = True,
                     marker_choice: str = 'box') -> SelectorBox:
    config = get_config()

    align_pipe = auto_op = None
    shift = 0

    try:
        if require_alignment:
            if config['align_pipe_path'] is None:
                raise ParamNotSet('align_pipe_path')
            if (config['coe'] is None) or (config['shift'] is None):
                print('Alignment must be run before linearization.')
                return

            shift = get_shift(which)

            auto_key = f"{which.split('_')[1]}_auto_op_path"
            auto_op = config[auto_key]

            if which.startswith('uv_'):
                align_pipe = v2v.load_pipeline(config['align_pipe_path'])

                align_idx, = (
                    i for i in align_pipe.nodes
                    if isinstance(align_pipe.nodes[i]['operator'],
                                  v2v.AutoAlign)
                )
                align_op = align_pipe.nodes[align_idx]['operator']
                align_op.coe = config['coe']
                align_op.output_size = config.image_size
                if isinstance(align_op, v2v.AutoTemporalAlign):
                    align_op.shift = config['shift']

        return SelectorBox(
            get_loader(which), shift, 50, box_color=(0, 255, 255),
            auto_op=auto_op, cache_path=get_cache_path(which),
            output_size=0.10, copy_from=copy_from, align_pipeline=align_pipe,
            marker_choice=marker_choice,
        )
    except ParamNotSet as err:
        print(f'Please specify {PARAM_CAPTIONS[err.args[0]].lower()}.')