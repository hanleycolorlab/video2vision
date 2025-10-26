from functools import partial
import os
import subprocess
import sys
from typing import Callable, Dict, Optional

from .config import get_config, PARAM_CAPTIONS

__all__ = [
    'choose_directory_dialog', 'choose_file_dialog', 'ConfigWindow',
    'save_file_dialog', 'show_config_window',
]


def _run_subprocess_dialog(dialog_type: str, prompt: Optional[str] = None, plural: bool = False):
    '''
    Run tkinter file dialog in a separate subprocess to avoid conflicts with Jupyter.
    '''
    script = f'''
import tkinter as tk
from tkinter import filedialog
import sys

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

prompt = {repr(prompt)}
dialog_type = {repr(dialog_type)}
plural = {repr(plural)}

result = None
if dialog_type == 'directory':
    result = filedialog.askdirectory(title=prompt)
elif dialog_type == 'openfile':
    if plural:
        result = filedialog.askopenfilenames(title=prompt)
        if result:
            result = '|||'.join(result)
    else:
        result = filedialog.askopenfilename(title=prompt)
elif dialog_type == 'savefile':
    result = filedialog.asksaveasfilename(title=prompt)

if result:
    print(result)
root.destroy()
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout.strip()
            if plural and dialog_type == 'openfile':
                return tuple(output.split('|||'))
            return output
        return None
    except subprocess.TimeoutExpired:
        print("Dialog timed out")
        return None
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return None


def choose_directory_dialog(prompt: Optional[str] = None):
    '''
    Creates an open file dialogue to select a directory.
    '''
    result = _run_subprocess_dialog('directory', prompt)

    # If subprocess failed or user cancelled, return None
    # The original code had a while loop requiring selection,
    # but that's not user-friendly if they want to cancel
    return result


def choose_file_dialog(prompt: Optional[str] = None, plural: bool = False):
    '''
    Creates an open file dialogue to select a file.

    Args:
        plural (bool): Whether to allow selecting multiple files. If true, this
        returns a List[str] instead of a string.
    '''
    return _run_subprocess_dialog('openfile', prompt, plural)


def save_file_dialog(prompt: Optional[str] = None) -> str:
    '''
    Creates a save file dialogue to select a file.
    '''
    result = _run_subprocess_dialog('savefile', prompt)
    return result


CONFIG_PROMPTS: Dict[str, Callable] = {
    'align_pipe_path': partial(
        choose_file_dialog, 'Select alignment pipeline',
    ),
    'vis_path': partial(
        choose_file_dialog, 'Select visible inputs',
    ),
    'uv_path': partial(
        choose_file_dialog, 'Select UV inputs',
    ),
    'uv_aligned_path': partial(
        choose_directory_dialog,
        'Select where to write aligned UV outputs',
    ),
    'animal_out_path': partial(
        choose_directory_dialog,
        'Select where to write animal perception outputs',
    ),
    'human_out_path': partial(
        choose_directory_dialog,
        'Select where to write linearized outputs',
    ),
    'vis_linearization_path': partial(
        choose_file_dialog,
        'Select image to use for visible linearization samples',
    ),
    'uv_linearization_path': partial(
        choose_file_dialog,
        'Select image to use for UV linearization samples',
    ),
    'linearization_values_path': partial(
        choose_file_dialog,
        'Select file containing linearization sample values',
    ),
    'camera_path': partial(
        choose_file_dialog,
        'Select file containing camera sensitivities',
    ),
    'vis_test_path': partial(
        choose_file_dialog,
        'Select file to use for visible test samples',
    ),
    'uv_test_path': partial(
        choose_file_dialog,
        'Select file to use for visible test samples',
    ),
    'test_values_path': partial(
        choose_file_dialog,
        'Select file containing test sample values',
    ),
    'linearization_auto_op_path': partial(
        choose_file_dialog,
        'Select linearization sample auto-locator, or select cancel for '
        'none',
    ),
    'test_auto_op_path': partial(
        choose_file_dialog,
        'Select test sample auto-locator, or select cancel for none',
    ),
    'animal_sensitivity_path': partial(
        choose_file_dialog,
        'Select file containing animal sensitivities',
    ),
    'sense_converter_path': partial(
        choose_file_dialog, 'Select sense converter',
    ),
    'save_align_pipe_path': partial(
        save_file_dialog, 'Save alignment pipeline as',
    ),
    'save_auto_op_path': partial(
        save_file_dialog, 'Save autolinearizer as',
    ),
    'reflectivity_path': partial(
        choose_file_dialog, 'Select reflectivity database',
    ),
    'save_converter_path': partial(
        save_file_dialog, 'Save sense converter as',
    ),
    'sample_record_path': partial(
        save_file_dialog, 'Save sample records as',
    ),
}


# We use global variables to store the window to avoid accidentally creating
# more than one of it.
_config_window = None


class ConfigWindow:
    '''
    This is a window displaying the current configuration options. It can be
    used to address the configuration directly, e.g.:

    .. code::
        pw = ParamsWindow()
        pw.shift = 1

    This will set shift to 1 in the global configuration object, and
    simultaneously update the display window. If this object is destroyed, a
    new one can be created and will retain the same status from the global
    :class:`Params` object.
    '''
    params = (
        'experiment_name',
        'align_pipe_path',
        'vis_path',
        'uv_path',
        'uv_aligned_path',
        'animal_out_path',
        'human_out_path',
        'vis_linearization_path',
        'uv_linearization_path',
        'linearization_values_path',
        'camera_path',
        'is_sony_camera',
        'vis_test_path',
        'uv_test_path',
        'test_values_path',
        'linearization_auto_op_path',
        'test_auto_op_path',
        'animal_sensitivity_path',
        'sense_converter_path',
    )

    def __init__(self):
        import tkinter as tk
        from tkinter import Toplevel, Label

        root = tk.Tk()
        root.withdraw()
        self.window = Toplevel(root)

        config = get_config()

        for row, k in enumerate(self.params, 1):
            label = Label(self.window, text=PARAM_CAPTIONS[k])
            label.grid(column=1, row=row)
            label = Label(self.window, text=config._label_text(k))
            label.grid(column=2, row=row)
            config._popup_labels[k] = label

        self.window.resizable(False, False)
        self.window.geometry(f'600x{len(self.params) * 22}')


def show_config_window(*args) -> ConfigWindow:
    global _config_window

    if _config_window is None:
        _config_window = ConfigWindow()

    return _config_window
