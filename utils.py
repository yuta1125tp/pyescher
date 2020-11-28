import numpy as np


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
