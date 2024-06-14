from functools import partial
from tkinter import filedialog, Label, Tk, Toplevel
from typing import Callable, Dict, Optional

from .config import get_config, PARAM_CAPTIONS

__all__ = [
    'choose_directory_dialog', 'choose_file_dialog', 'ConfigWindow',
    'save_file_dialog', 'show_config_window',
]


def choose_directory_dialog(prompt: Optional[str] = None):
    '''
    Creates an open file dialogue to select a directory.
    '''
    Tk().withdraw()
    out = None

    while not out:
        out = filedialog.askdirectory(title=prompt)

    return out


def choose_file_dialog(prompt: Optional[str] = None, plural: bool = False):
    '''
    Creates an open file dialogue to select a file.

    Args:
        plural (bool): Whether to allow selecting multiple files. If true, this
        returns a List[str] instead of a string.
    '''
    Tk().withdraw()
    ask = filedialog.askopenfilenames if plural else filedialog.askopenfilename
    return ask(title=prompt)


def save_file_dialog(prompt: Optional[str] = None) -> str:
    '''
    Creates a save file dialogue to select a file.
    '''
    Tk().withdraw()
    out = None

    while not out:
        out = filedialog.asksaveasfilename(title=prompt)

    return out


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


class ConfigWindow(Toplevel):
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
        root = Tk()
        root.withdraw()
        super().__init__(root)

        config = get_config()

        for row, k in enumerate(self.params, 1):
            label = Label(self, text=PARAM_CAPTIONS[k])
            label.grid(column=1, row=row)
            label = Label(self, text=config._label_text(k))
            label.grid(column=2, row=row)
            config._popup_labels[k] = label

        self.resizable(False, False)
        self.geometry(f'600x{len(self.params) * 22}')


def show_config_window(*args) -> ConfigWindow:
    global _config_window

    if _config_window is None:
        _config_window = ConfigWindow()

    return _config_window
