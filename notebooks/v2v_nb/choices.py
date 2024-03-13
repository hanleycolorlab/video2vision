'''
Provides notebook widgets for entering selections.
'''
from typing import Callable, Optional

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
    'BoolBox', 'ConfigBox', 'IntBox', 'PathBox', 'SimpleButton', 'StringBox',
]


TEXT_WIDTH = '420px'
BUTTON_WIDTH = '80px'
LABEL_HEIGHT = '20px'
WIDGET_HEIGHT = '30px'


class LabeledBox(widgets.VBox):
    def __init__(self, key: str, favorite: widgets.Widget,
                 button: Optional[widgets.Widget] = None):
        config = get_config()

        if config._nb_labels[key] is not None:
            favorite = config._nb_labels[key]
        else:
            config._nb_labels[key] = favorite

        label = widgets.Label(PARAM_CAPTIONS[key])
        if button is None:
            button = widgets.Label(
                '',
                layout=widgets.Layout(width=BUTTON_WIDTH,
                                      height=WIDGET_HEIGHT),
            )
        hbox = widgets.HBox((favorite, button))
        super().__init__((label, hbox))

        self.key = key
        self.favorite.continuous_update = False
        self.favorite.observe(self.callback, 'value')

    def callback(self, widget):
        config = get_config()
        config[self.key] = self.value

    @property
    def favorite(self) -> widgets.Widget:
        return self.children[1].children[0]

    @property
    def value(self):
        return self.favorite.value

    @value.setter
    def value(self, v):
        self.favorite.value = v


class BoolBox(LabeledBox):
    def __init__(self, key: str):
        config = get_config()
        checkbox = widgets.Checkbox(
            value=config._values[key] or False,
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__(key, checkbox)


class IntBox(LabeledBox):
    def __init__(self, key: str):
        config = get_config()

        if config._values[key] is None:
            value = ''
        else:
            value = str(config._values[key])

        textbox = widgets.Text(
            value=value,
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__(key, textbox)

    def callback(self, widget):
        config = get_config()
        if self.value == '':
            config[self.key] = None
        else:
            try:
                config[self.key] = int(self.value)
            except ValueError:
                self.value = str(config[self.key])


class PathBox(LabeledBox):
    def __init__(self, key: str, kind: str = 'openfile'):
        config = get_config()
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
        super().__init__(key, textbox, button)

        button.on_click(self.dialog_box)

    def dialog_box(self, button):
        out = CONFIG_PROMPTS[self.key]()

        if out:
            self.favorite.value = out


class StringBox(LabeledBox):
    def __init__(self, key: str):
        config = get_config()
        textbox = widgets.Text(
            value=(config._values[key] or ''),
            disabled=False,
            layout=widgets.Layout(width=TEXT_WIDTH, height=WIDGET_HEIGHT),
        )
        super().__init__(key, textbox)


BOX_TYPES = {
    'bool': BoolBox, 'int': IntBox, 'path': PathBox, 'string': StringBox,
}


class ConfigBox(widgets.VBox):
    def __init__(self, *keys: str, show_buttons: bool = False):
        contents = [BOX_TYPES[PARAM_TYPES[k]](k) for k in keys]
        if show_buttons:
            buttons = (
                SimpleButton('Show Config', show_config_window),
                SimpleButton('Save Defaults', save_defaults),
                SimpleButton('Clear All', clear_all),
            )
            contents += [widgets.HBox(buttons)]
        super().__init__(contents)


class SimpleButton(widgets.Button):
    def __init__(self, description: str, action: Callable):
        super().__init__(description=description)
        self.action = action
        self.on_click(self._click)

    def _click(self, b):
        self.action()
