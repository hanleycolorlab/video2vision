'''
Provides notebook widgets for entering selections.
'''
import ipywidgets as widgets

from .config import (
    clear_all,
    get_config,
    PARAM_CAPTIONS,
    PARAM_TYPES,
    save_defaults
)
from .popup import CONFIG_PROMPTS, show_config_window

__all__ = [
    'BoolBox', 'ConfigBox', 'IntBox', 'PathBox', 'StringBox',
    'SaveDefaultsButton', 'ShowConfigButton'
]


LABEL_WIDTH = '120px'
TEXT_WIDTH = '300px'
BUTTON_WIDTH = '80px'
WIDGET_HEIGHT = '30px'


class BoolBox(widgets.HBox):
    def __init__(self, key: str):
        self.key = key

        config = get_config()
        if config._nb_labels[key] is not None:
            raise RuntimeError(
                f'Multiple notebook labels registered for {key}'
            )

        label = widgets.Label(
            PARAM_CAPTIONS[key],
            layout=widgets.Layout(width=LABEL_WIDTH, height=WIDGET_HEIGHT),
        )
        checkbox = widgets.Checkbox(
            value=config._values[key] or False,
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        # Used to keep alignment consistent with PathBox
        whitespace = widgets.Label(
            '',
            layout=widgets.Layout(width=BUTTON_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__((label, checkbox, whitespace))

        config._nb_labels[key] = checkbox
        checkbox.continuous_update = False
        checkbox.observe(self.callback, 'value')

    def callback(self, checkbox):
        config = get_config()
        config[self.key] = self.children[1].value


class IntBox(widgets.HBox):
    def __init__(self, key: str):
        self.key = key

        config = get_config()
        if config._nb_labels[key] is not None:
            raise RuntimeError(
                f'Multiple notebook labels registered for {key}'
            )

        label = widgets.Label(
            PARAM_CAPTIONS[key],
            layout=widgets.Layout(width=LABEL_WIDTH, height=WIDGET_HEIGHT),
        )
        textbox = widgets.IntText(
            value=config._values[key],
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        # Used to keep alignment consistent with PathBox
        whitespace = widgets.Label(
            '',
            layout=widgets.Layout(width=BUTTON_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__((label, textbox, whitespace))

        config._nb_labels[key] = textbox
        textbox.continuous_update = False
        textbox.observe(self.callback, 'value')

    def callback(self, textbox):
        config = get_config()
        config[self.key] = self.children[1].value


class StringBox(widgets.HBox):
    def __init__(self, key: str):
        self.key = key

        config = get_config()
        if config._nb_labels[key] is not None:
            raise RuntimeError(
                f'Multiple notebook labels registered for {key}'
            )

        label = widgets.Label(
            PARAM_CAPTIONS[key],
            layout=widgets.Layout(width=LABEL_WIDTH, height=WIDGET_HEIGHT),
        )
        textbox = widgets.Text(
            value=(config._values[key] or ''),
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        # Used to keep alignment consistent with PathBox
        whitespace = widgets.Label(
            '',
            layout=widgets.Layout(width=BUTTON_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__((label, textbox, whitespace))

        config._nb_labels[key] = textbox
        textbox.continuous_update = False
        textbox.observe(self.callback, 'value')

    def callback(self, textbox):
        config = get_config()
        config[self.key] = self.children[1].value


class PathBox(widgets.HBox):
    def __init__(self, key: str, kind: str = 'openfile'):
        self.key = key

        config = get_config()
        if config._nb_labels[key] is not None:
            raise RuntimeError(
                f'Multiple notebook labels registered for {key}'
            )

        label = widgets.Label(
            PARAM_CAPTIONS[key],
            layout=widgets.Layout(width=LABEL_WIDTH, height=WIDGET_HEIGHT),
        )
        textbox = widgets.Text(
            value=(config._values[key] or ''),
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        button = widgets.Button(
            description='Browse',
            disabled=False,
            layout=widgets.Layout(width=BUTTON_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__((label, textbox, button))

        config._nb_labels[key] = textbox
        textbox.continuous_update = False
        textbox.observe(self.callback, 'value')
        button.on_click(self.dialog_box)

    def callback(self, textbox):
        config = get_config()
        config[self.key] = self.children[1].value

    def dialog_box(self, button):
        out = CONFIG_PROMPTS[self.key]()

        if out:
            self.children[1].value = out


BOX_TYPES = {
    'bool': BoolBox, 'int': IntBox, 'path': PathBox, 'string': StringBox,
}


class ConfigBox(widgets.VBox):
    def __init__(self, *keys: str, show_buttons: bool = False):
        contents = [BOX_TYPES[PARAM_TYPES[k]](k) for k in keys]
        if show_buttons:
            buttons = (
                ShowConfigButton(), SaveDefaultsButton(), ClearAllButton()
            )
            contents += [widgets.HBox(buttons)]
        super().__init__(contents)


class ClearAllButton(widgets.Button):
    def __init__(self):
        super().__init__(description='Clear All')
        self.on_click(self._click)

    def _click(self, b):
        clear_all()


class SaveDefaultsButton(widgets.Button):
    def __init__(self):
        super().__init__(description='Save Defaults')
        self.on_click(self._click)

    def _click(self, b):
        save_defaults()


class ShowConfigButton(widgets.Button):
    def __init__(self):
        super().__init__(description='Show Config')
        self.on_click(self._click)

    def _click(self, b):
        show_config_window()
