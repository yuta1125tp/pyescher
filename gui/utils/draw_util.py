# pylint: disable=
# pylint: disable=E1101 # no-member


import logging
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from gui.data_class.CPoint import CPoint
from gui.data_class.CPoints import CPoints
from gui.utils.points_util import get_rect_xy


def putlabel(
    img: np.ndarray,
    points: CPoints,
    label: List[str] = ["left top", "right top", "right bottom", "left bottom"],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    color = (255, 255, 255)
    thickness = 1
    buf = 5
    for i in range(min(len(points), len(label))):
        p = points[i]
        cv2.putText(
            img,
            label[i],
            (p.x + buf, p.y - buf),
            font,
            font_size,
            color,
            thickness,
            cv2.LINE_AA,
        )


def drawrect(
    img: np.ndarray, points: CPoints, color=(0, 255, 0), thickness=-1, **kwargs
) -> None:
    if len(points) != 2:
        # print('len(points)!=2',len(points))
        return None
    xmin, xmax, ymin, ymax = get_rect_xy(points)
    # logger = logging.getLogger(__name__)
    # logger.info(f"{xmin}, {xmax}, {ymin}, {ymax}")
    # logger.info(f"{type(xmin)}, {type(xmax)}, {type(ymin)}, {type(ymax)}")
    # logger.info(f"{color}, {thickness}")
    # logger.info(
    #     f"{type(color)}, {type(thickness)}, {type(ymin)}, {type(ymax)}")
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness, **kwargs)


def drawmesh(img: np.ndarray) -> None:
    color = (0, 0, 255)
    thickness = 1
    freq = 0.10  # 間隔
    height, width = img.shape[:2]
    for y in range(0, height, int(height * freq)):
        cv2.line(img, (0, y), (width - 1, y), color, thickness)
    for x in range(0, width, int(width * freq)):
        cv2.line(img, (x, 0), (x, height - 1), color, thickness)


def drawpoints(img: np.ndarray, points: CPoints) -> None:
    color = (0, 0, 255)
    thickness = 3
    size = 3
    for _, p in enumerate(points):
        cv2.circle(img, (p.x, p.y), size, color, thickness)
