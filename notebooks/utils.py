from typing import Tuple

import cv2
from PIL import Image

__all__ = ['get_size']


def get_size(path: str) -> Tuple[int, int]:
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
        return Image.open(path).size
