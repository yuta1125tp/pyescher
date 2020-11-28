import logging
from typing import List

import cv2
import numpy as np

from gui.data_class import CEdge, CPoint, CPoints
from gui.utils.points_util import nearest_point_idx, xy_locate_in_rect
from gui.utils.draw_util import drawmesh, drawpoints, drawrect, putlabel

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

current_target_idx_rect = -3  # -1:move rect, -2:move edge
prev_target_idx_rect = -1
# edgeで修正するときにつかう
modi_edge_idx = 0

active_window_idx = -1
active_stitch_tgt_idx = 0

last_clicked_point = [-1, -1]

# 既存の点をクリックした判定のためのしきい値
threshold = 10
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


def on_mouse_event(event, x, y, flag, params):
    logger = logging.getLogger(__name__)

    global current_target_idx_rect, prev_target_idx_rect, last_clicked_point
    global active_window_idx, active_stitch_tgt_idx, state_dict, modi_edge_idx

    active_window_idx = 1
    # key_points = state_dict["key_points"]
    rect_points = state_dict["rect_points"]

    [
        wname_tv,
        tv_img,
    ] = params
    tv_img_canvas = np.copy(tv_img)

    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリック
        npidx = nearest_point_idx(x, y, rect_points)
        if len(rect_points) == 2:
            rect_4points = CPoints(
                [
                    rect_points[0],
                    [rect_points[1].x, rect_points[0].y],
                    rect_points[1],
                    [rect_points[0].x, rect_points[1].y],
                ]
            )
            edge_dist_list = dist_xy_to_edges(x, y, rect_4points)
        else:
            edge_dist_list = []
        logger.info(edge_dist_list)
        if npidx >= 0 and rect_points[npidx].dist(x, y) < threshold:
            # 最も近い点がしきい値以内ならそれを興味のある点にして移動できるようにする
            current_target_idx_rect = npidx
        elif len(edge_dist_list) and np.any(np.asarray(edge_dist_list) < threshold):
            modi_edge_idx = np.argmin(edge_dist_list)
            current_target_idx_rect = -2  # move edge
        elif xy_locate_in_rect(x, y, rect_points):
            current_target_idx_rect = -1  # move rect
        else:
            # それ以外なら点を新たに追加
            current_target_idx_rect = len(rect_points)
            rect_points.add(x, y)
            drawpoints(tv_img_canvas, rect_points)
            drawrect(tv_img_canvas, rect_points)
            cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_target_idx_rect >= 0:
            # 興味のある点があれば位置を更新
            rect_points[current_target_idx_rect].setxy(x, y)
        elif current_target_idx_rect == -1:
            # move rect
            diff = [x - last_clicked_point[0], y - last_clicked_point[1]]
            for point in rect_points:
                point.setxy(point.x + diff[0], point.y + diff[1])
        elif current_target_idx_rect == -2:
            # move edge
            diff = [x - last_clicked_point[0], y - last_clicked_point[1]]
            if modi_edge_idx == 0:
                rect_points[0]._y += diff[1]
            elif modi_edge_idx == 1:
                rect_points[1]._x += diff[0]
            elif modi_edge_idx == 2:
                rect_points[1]._y += diff[1]
            elif modi_edge_idx == 3:
                rect_points[0]._x += diff[0]

        drawpoints(tv_img_canvas, rect_points)
        drawrect(tv_img_canvas, rect_points)
        cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    elif event == cv2.EVENT_LBUTTONUP:
        # 興味のある点を前回の興味のある点に登録して-1で上書き
        prev_target_idx_rect = current_target_idx_rect
        current_target_idx_rect = -3
        drawpoints(tv_img_canvas, rect_points)
        drawrect(tv_img_canvas, rect_points)
        cv2.imshow(wname_tv, tv_img_canvas.astype(np.uint8))

    last_clicked_point = [x, y]


import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
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
        key_points=CPoints(),
        rect_points=CPoints(),
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
            if active_window_idx == 0:
                state_dict["key_points"].rm(prev_target_idx)
                prev_target_idx = -1
            elif active_window_idx == 1:
                state_dict["rect_points"].rm(prev_target_idx_rect)
                prev_target_idx_rect = -1
            else:
                logger.info(
                    f"d:del active_windows_idx:{active_window_idx} does not supported."
                )
    cv2.destroyAllWindows()
    # if label_file is not None:
    #     logger.info(f'save label as "{args.label_file}"')
    #     save_label_file(stat_dict, label_file)


if __name__ == "__main__":
    main()
