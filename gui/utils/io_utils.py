import collections.abc
import copy
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from gui.data_class import CPoints


class NumpyEncoder(json.JSONEncoder):
    """[NumPy array is not JSON serializable](https://stackoverflow.com/a/47626762)"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, CPoints):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_label_file(
    load_from: Path, cpoint_keys=["thumbnail_rect", "key_points", "rect_points"]
) -> Dict:
    loaded = json.loads(load_from.read_text())
    for key1 in loaded:
        for cpoint_key in cpoint_keys:
            if cpoint_key in loaded[key1]:
                loaded[key1][cpoint_key] = CPoints(loaded[key1][cpoint_key])
    return loaded


def update(update_this: Dict, update_by: Dict) -> Dict:
    """recursive update dict
    [](https://stackoverflow.com/a/3233356)"""
    for key, val in update_by.items():
        if isinstance(val, collections.abc.Mapping):
            update_this[key] = update(update_this.get(key, {}), val)
        else:
            update_this[key] = val
    return update_this


def save_label_file(
    label: Dict, save_as: Path, block_list=["image", "thumbnail"]
) -> None:

    label_cp = copy.deepcopy(label)
    for key in label_cp:
        for block_word in block_list:
            if block_word in label_cp[key]:
                del label_cp[key][block_word]

    if len(str(save_as.parent)):
        save_as.parent.mkdir(exist_ok=True, parents=True)

    save_as.write_text(json.dumps(label_cp, cls=NumpyEncoder, indent=2))
