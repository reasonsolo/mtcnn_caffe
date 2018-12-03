import os
import sys
from glob import glob


def iterate_300w(path, functor):
    """
    functor(img_path, bboxes)
    """
    for root, dirs, files in os.walk(path):
        pass
