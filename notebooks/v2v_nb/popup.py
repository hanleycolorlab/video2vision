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
        out = filedialog.askdirectory(title='video2vision', message=prompt)

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
    return ask(title='video2vision', message=prompt)


def save_file_dialog(prompt: Optional[str] = None) -> str:
    '''
    Creates a save file dialogue to select a file.
    '''
    Tk().withdraw()
    out = None

    while not out:
        out = filedialog.asksaveasfilename(
            title='video2vision', message=prompt
        )

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
    def __init__(self):
        root = Tk()
        root.withdraw()
        super().__init__(root)

        config = get_config()

        for row, (k, caption) in enumerate(PARAM_CAPTIONS.items(), 1):
            label = Label(self, text=config._label_text(k))
            label.grid(column=1, row=row)
            config._labels[k] = label

        self.resizable(False, False)
        self.geometry(f'400x{len(config._labels) * 22}')


def show_config_window(*args) -> ConfigWindow:
    global _config_window

    if _config_window is None:
        _config_window = ConfigWindow()

    return _config_window
