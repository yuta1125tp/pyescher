import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from gui.data_class.CPoint import CPoint
from gui.data_class.CPoints import CPoints


def nearest_point_idx(x: int, y: int, points: CPoints) -> int:
    """find nearest point idx."""
    if not len(points):
        return -1
    tmp = [p.dist(x, y) for p in points]
    return np.argmin(tmp)


def show_points(points: CPoints) -> None:
    logger = logging.getLogger(__name__)
    for i, p in enumerate(points):
        logger.info("[{}/{}] ( {}, {} )".format(i, len(points), p.x, p.y))


def get_rect_xy(points: CPoints) -> Tuple[int]:
    if type(points[0]) == CPoint:
        xmin = min(points[0].x, points[1].x)
        xmax = max(points[0].x, points[1].x)
        ymin = min(points[0].y, points[1].y)
        ymax = max(points[0].y, points[1].y)
    else:
        xmin = min(points[0][0], points[1][0])
        xmax = max(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        ymax = max(points[0][1], points[1][1])
    return xmin, xmax, ymin, ymax


def read_points(points: CPoints, inputfilename: Path):
    with inputfilename.open("r") as fin:
        lines = [
            [int(ee) for ee in e.split() if len(ee)]
            for e in fin.read().split("\n")
            if len(e)
        ]
    points.clear()
    for line in lines:
        points.add(*line)


def write_points(points: CPoints, outfilename: Path):
    with outfilename.open("w") as fout:
        for p in points:
            line = " ".join([str(e) for e in [p.x, p.y]])
            fout.write(line + "\n")


def xy_locate_in_rect(x: float, y: float, points: CPoints):
    if len(points) != 2:
        # print('points have to be len()==2')
        return False
    xmin, xmax, ymin, ymax = get_rect_xy(points)
    return xmin <= x and x < xmax and ymin <= y and y < ymax
