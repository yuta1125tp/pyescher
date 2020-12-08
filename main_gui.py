import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from gui.data_class import CEdge, CPoint, CPoints
from gui.utils.draw_util import drawmesh, drawpoints, drawpoly, drawrect, putlabel
from gui.utils.io_utils import load_label_file, save_label_file
from gui.utils.points_util import nearest_point_idx, xy_locate_in_rect


class ListedColormapCV2:
    def __init__(self, name):
        _cmap = plt.get_cmap(name)
        self._color_list = [
            tuple([int(elm * 255) for elm in color[::-1]]) for color in _cmap.colors
        ]

    @property
    def colors(self):
        return self._color_list

    def __call__(self, index):
        return self._color_list[index]


cmap = ListedColormapCV2("tab10")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(levelname)s : %(filename)s #%(lineno)d :%(message)s",
)
logger = logging.getLogger()


# == dirty global variables ==
state_dict = dict()

toolname = "PyEscherGUI"

current_target_idx = -1
prev_target_idx = -1

current_target_idx = -3  # -1:move rect, -2:move edge
prev_target_idx_rect = -1
# edgeで修正するときにつかう
modi_edge_idx = 0

active_window_idx = -1
active_stitch_tgt_idx = 0

last_clicked_point = [-1, -1]

# 既存の点をクリックした判定のためのしきい値
threshold = 10

# 1 : center_inner
# 2 : center_outter
# 3 : raius
# 4 : rect
# 5 : scale
point_type_dict = dict(
    [
        [1, "center_inner"],
        [2, "center_outter"],
        [3, "radius"],
        [4, "rect"],
        [5, "scale"],
    ]
)
current_point_type_idx = 1
# == ~dirty global variables ==


def dist_xy_to_edges(x: int, y: int, points: CPoints) -> List[float]:
    very_far = 1e8
    logger = logging.getLogger(__name__)
    dist_list = []
    num_points = len(points)
    if num_points < 2:
        return []
    for idx in range(num_points):
        if idx == num_points - 1:
            pointA = points[idx]
            pointB = points[0]
        else:
            pointA = points[idx]
            pointB = points[idx + 1]

        logger.info(idx)
        logger.info(num_points)
        logger.info(pointA)
        logger.info(pointB)
        edgeAB = CEdge(pointA, pointB)
        if edgeAB.on_edge(x, y):
            dist = edgeAB.dist(x, y)
        else:
            dist = very_far
        dist_list.append(dist)
    return dist_list


def draw_points_all(img: np.ndarray, key_points_dict: dict):
    for i, key in enumerate(key_points_dict):
        key_points = key_points_dict[key]
        color = cmap(i % len(cmap.colors))
        print(color)
        drawpoints(img, key_points, color=color)
        drawpoly(img, key_points, color=color)


def on_mouse_event(event, x, y, flag, params):
    logger = logging.getLogger(__name__)

    global current_target_idx, prev_target_idx, last_clicked_point
    global active_window_idx, active_stitch_tgt_idx, state_dict, modi_edge_idx
    global current_point_type_idx

    active_window_idx = 1
    key_points_dict = state_dict["key_points_dict"]
    key_points = key_points_dict[f"key_points_{current_point_type_idx}"]

    [
        wname_tv,
        tv_img,
    ] = params
    tv_img_canvas = np.copy(tv_img)

    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリック
        npidx = nearest_point_idx(x, y, key_points)
        edge_dist_list = dist_xy_to_edges(x, y, key_points)
        logger.info(edge_dist_list)

        if npidx >= 0 and key_points[npidx].dist(x, y) < threshold:
            # 最も近い点がしきい値以内ならそれを興味のある点にして移動できるようにする
            current_target_idx = npidx
        elif len(edge_dist_list) and np.any(np.asarray(edge_dist_list) < threshold):
            modi_edge_idx = np.argmin(edge_dist_list) + 1
            # 最も近傍のエッジに点を追加
            key_points.insert(modi_edge_idx, x, y)
            current_target_idx = modi_edge_idx
            draw_points_all(tv_img_canvas, key_points_dict)
            cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))
        elif xy_locate_in_rect(x, y, key_points):
            current_target_idx = -1  # move poly
        else:
            # それ以外なら点を新たに追加
            current_target_idx = len(key_points)
            key_points.add(x, y)
            draw_points_all(tv_img_canvas, key_points_dict)
            cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    if event == cv2.EVENT_RBUTTONDOWN:  # 右クリック
        logger.info("right click")
        # 中心に関してどうのこうの

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_target_idx >= 0:
            # 興味のある点があれば位置を更新
            key_points[current_target_idx].setxy(x, y)
        elif current_target_idx == -1:
            # move rect
            diff = [x - last_clicked_point[0], y - last_clicked_point[1]]
            for point in key_points:
                point.setxy(point.x + diff[0], point.y + diff[1])

        draw_points_all(tv_img_canvas, key_points_dict)
        cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    elif event == cv2.EVENT_LBUTTONUP:
        # 興味のある点を前回の興味のある点に登録して-1で上書き
        prev_target_idx = current_target_idx
        current_target_idx = -3
        draw_points_all(tv_img_canvas, key_points_dict)
        cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    last_clicked_point = [x, y]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("--label_file", type=Path, default=Path("./tmp.json"))
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(args)

    # # どのウインドウが選択中か
    # global active_window_idx
    # # どの画像を継ぎ接ぎしているか
    # global active_stitch_tgt_idx

    # 前回編集した点の通し番号
    global prev_target_idx
    global prev_target_idx_rect
    # 状態の辞書
    global state_dict

    img = cv2.imread(str(args.input_image))

    state_dict = dict(
        # image=input_imgs[idx],
        # image_shape=input_imgs[idx].shape,
        key_points_dict=defaultdict(lambda: CPoints()),
    )
    if args.label_file.exists():
        state_dict.update(
            load_label_file(
                args.label_file,
                cpoint_keys=[f"key_points_{i}" for i in range(10)],
            )
        )

    wname_main = "{} (Main)".format(toolname)
    cv2.namedWindow(wname_main)  # , cv2.WINDOW_NORMAL)  # ウインドサイズ可変
    cv2.setMouseCallback(wname_main, on_mouse_event, [wname_main, img])
    cv2.imshow(wname_main, img.astype(np.uint8))

    esc_code = 27

    while 1:
        key = cv2.waitKey(1000)
        if key == esc_code:
            # esc
            break
        elif key == ord("d"):
            # del last modi point
            logger.info(
                f"Dell last modified point: active_window_idx={active_window_idx}"
            )
            if active_window_idx == 1:
                state_dict["key_points"].rm(prev_target_idx)
                prev_target_idx = -1
            else:
                logger.info(
                    f"d:del active_windows_idx:{active_window_idx} does not supported."
                )
        elif key in [ord(f"{i}") for i in range(10)]:
            # del last modi point
            current_point_type_idx = key - 48
            point_type_str = point_type_dict.get(current_point_type_idx, "None")
            logger.info(
                f"current_point_type_idx: {current_point_type_idx}: {point_type_str}"
            )

    cv2.destroyAllWindows()

    if args.label_file is not None:
        logger.info(f'save label as "{args.label_file}"')
        save_label_file(state_dict, args.label_file)


if __name__ == "__main__":
    main()
