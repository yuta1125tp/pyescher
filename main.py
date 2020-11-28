# coding:utf-8
import argparse
import itertools
import json
import logging
import pprint
from pathlib import Path

import cv2
import numpy as np
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(levelname)s : %(filename)s #%(lineno)d :%(message)s",
)
logger = logging.getLogger()


def in_range(value, min_v, max_v, is_closed_sectoin=[True, True]):
    if is_closed_sectoin[0]:
        if min_v <= value:
            pass
        else:
            return False
    else:
        if min_v < value:
            pass
        else:
            return False
    if is_closed_sectoin[1]:
        if value <= max_v:
            pass
        else:
            return False
    else:
        if value < max_v:
            pass
        else:
            return False
    return True


def xy_to_z(xy: np.ndarray) -> np.ndarray:
    """return [x,y] * [1,1j]"""
    return np.dot(xy, np.array([1, 1j]))


def z_to_xy(z: np.complex) -> np.ndarray:
    """"""
    return np.stack([z.real, z.imag]).T


def is_inside_circle(x, y, a, b, c, cx=0, cy=0):
    return a * (x - cx) ** 2 + b * (y - cy) ** 2 - c ** 2 < 0


def is_outside_circle(x, y, a, b, c, cx=0, cy=0):
    return not is_inside_circle(x, y, a, b, c, cx, cy)


def main():
    """"""
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("config_json", type=Path)
    parser.add_argument("--concentric", action="store_true")
    parser.add_argument("--inverse_mapping", action="store_true")
    parser.add_argument("--bg_color", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--tmp_dir", type=Path, default="tmp")
    parser.add_argument("--min_idx", type=int, default=0)
    parser.add_argument("--max_idx", type=int, default=8)

    args = parser.parse_args()

    args.tmp_dir.mkdir(exist_ok=True, parents=True)

    img = cv2.imread(str(args.input_image))
    # image_scale = 1.0
    # img = cv2.resize(img, None, fx=image_scale, fy=image_scale)

    # img = cv2.resize(
    #     img,
    #     (200, 200),
    # )
    cv2.imwrite(str(args.tmp_dir / "img_resized.jpg"), img)

    config_json = json.loads(args.config_json.read_text())
    logger.info("args")
    for arg in vars(args):
        logger.info("{0:<20}: {1}".format(arg, getattr(args, arg)))

    logger.info("config")
    logger.info(
        pprint.pformat(
            config_json,
            indent=2,
            width=1,
        )
    )

    height, width = img.shape[:2]

    canvas_height, canvas_width = height, width
    base_color = args.bg_color
    canvas = np.zeros([canvas_height, canvas_width, 3], dtype=np.uint8) + base_color

    center = np.asarray([width / 2, height / 2])
    logger.info(f"{height = }")
    logger.info(f"{width  = }")
    logger.info(f"{center = }")
    if not args.inverse_mapping:
        x_range = range(width)
        y_range = range(height)
    else:
        x_range = range(canvas_width)
        y_range = range(canvas_height)

    xy_list = np.array(list(itertools.product(x_range, y_range)))
    xy_list_shift = xy_list - center

    # prepare is_target_area_mask
    # init all True
    is_target_area_mask = np.zeros([height, width], dtype=bool)
    C_L = config_json["C_L"]
    C_S = config_json["C_S"]
    r_1 = config_json["r_1"]
    r_2 = config_json["r_2"]
    a_s = 1
    b_s = 1
    c_s = r_1
    a_l = 1
    b_l = 1
    c_l = r_2

    for row, col in itertools.product(range(height), range(width)):
        if is_inside_circle(row, col, a_s, b_s, c_s, C_S[0], C_S[1]):
            # inner false
            is_target_area_mask[col, row] = False
        elif is_outside_circle(row, col, a_l, b_l, c_l, C_L[0], C_L[1]):
            # outer false
            is_target_area_mask[col, row] = False
        else:
            is_target_area_mask[col, row] = True

    cv2.imwrite(
        str(args.tmp_dir / "is_target_area_mask.jpg"), is_target_area_mask * 255
    )

    img2 = np.zeros_like(img)
    img2[is_target_area_mask] = img[is_target_area_mask]
    cv2.imwrite(str(args.tmp_dir / "img_masked.jpg"), img2)

    scale1 = 100

    # 角度のオフセット
    theta = 0

    args.concentric = False
    if args.concentric:
        # 同心円
        alpha = 0
    else:
        # 螺旋
        alpha = np.arctan(np.log(r_1 / r_2) / (2 * np.pi))

    f = (2 * np.pi) / (2 * np.pi - theta) * np.cos(alpha)
    beta = f * np.exp(1j * alpha)

    # src_xy_ = np.array(list(itertools.product(x_range, y_range)))
    # dst_xy_ = np.array(list(itertools.product(x_range, y_range)))
    src_xy_ = np.zeros_like(xy_list)
    dst_xy_ = np.zeros_like(xy_list)

    eps = 1e-6

    idx_list = range(args.min_idx, args.max_idx)
    for idx in tqdm.tqdm(idx_list):
        if not args.inverse_mapping:
            # 順方向
            z_src = xy_to_z(xy_list_shift + eps)
            log_z = beta * (np.log(z_src / r_1) - idx * (np.log(r_1) - np.log(r_2)))
            z_dst = np.exp(log_z)
            # z_dst = (z_src / r_1) ** beta
            xy_list2_shift = z_to_xy(z_dst)
            xy_list2 = xy_list2_shift + center
            xy_list2 = xy_list2.astype(np.int)
            src_xy = xy_list
            src_xy = (xy_list_shift + center).astype(np.int)
            # dst_xy = xy_list2
            dst_xy = np.clip(xy_list2, 0, min(canvas_height, canvas_width) - 1)
        else:
            # 逆方向
            z_dst = xy_to_z(xy_list_shift + eps)
            partA = np.log(z_dst) / beta
            partB = idx * (np.log(r_1) - np.log(r_2))
            partC = np.log(r_1)
            log_z = partA + partB + partC
            # log_z = (
            #     np.log(z_dst) / beta + idx * (np.log(r_1) - np.log(r_2)) + np.log(r_1)
            # )
            if 1:
                z_src = np.exp(log_z)
                xy_list2_shift = z_to_xy(z_src)
                xy_list2 = xy_list2_shift + center
            else:
                xy_list2 = z_to_xy(log_z) * scale1
            xy_list2 = xy_list2.astype(np.int)

            src_xy = np.clip(xy_list2, 0, min(height, width) - 1)
            dst_xy = xy_list

        for idx, (p_src, p_dst) in enumerate(zip(src_xy, dst_xy)):
            if not is_target_area_mask[p_src[1], p_src[0]]:
                continue
            else:
                src_xy_[idx] = p_src
                dst_xy_[idx] = p_dst

    for idx, (p_src, p_dst) in enumerate(zip(src_xy_, dst_xy_)):
        canvas[p_dst[1], p_dst[0], :] = img[p_src[1], p_src[0], :]

    cv2.imwrite(str(args.tmp_dir / "result.jpg"), canvas)
    cv2.imwrite(str(args.output_image), canvas)
    return 0


if __name__ == "__main__":
    main()
