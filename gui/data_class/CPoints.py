# coding: utf-8
""""""

from typing import Callable, Dict, List, Tuple

import numpy as np

from .CPoint import CPoint
from .CEdge import CEdge


class CPoints(object):
    def __init__(self, point_list: List[Tuple[int, int]] = None):
        if point_list is None:
            self.point_list = []
        elif isinstance(point_list, list):
            assert all(len(p) == 2 for p in point_list)
            self.point_list = [
                elm if isinstance(elm, CPoint) else CPoint(*elm) for elm in point_list
            ]
        else:
            raise RuntimeError()
        self._iter_i = 0

    def __len__(self):
        return len(self.point_list)

    def __str__(self):
        res = self.__class__.__name__
        for p in self.point_list:
            res += str(p)
        return res

    def __iter__(self):
        # next()はselfが実装してるのでそのままselfを返す
        return self

    def __delitem__(self, key):
        del self.point_list[key]

    def __getitem__(self, key):
        return self.point_list[key]

    def __setitem__(self, key, value):
        self.point_list[key] = value

    def __next__(self):
        if self._iter_i == len(self.point_list):
            self._iter_i = 0
            raise StopIteration()
        value = self.point_list[self._iter_i]
        self._iter_i += 1
        return value

    def add(self, x, y):
        self.point_list.append(CPoint(x, y))

    def insert(self, index, x, y):
        self.point_list.insert(index, CPoint(x, y))

    def rm(self, idx):
        if 0 <= idx and idx < len(self.point_list):
            del self.point_list[idx]

    def clear(self):
        self.point_list = []

    def as_numpy(self):
        return np.asarray(self.tolist())

    def tolist(self):
        """numpy.ndarray like interface"""
        return [
            point.tolist() if isinstance(point, CPoint) else point
            for point in self.point_list
        ]

    def toedges(self, loop_close=False):
        if len(self) < 2:
            return []
        if loop_close:
            return [
                CEdge(p0, p1)
                for p0, p1 in zip(
                    self.point_list, self.point_list[1:] + self.point_list[:1]
                )
            ]
        else:
            return [
                CEdge(p0, p1)
                for p0, p1 in zip(self.point_list[:-1], self.point_list[1:])
            ]


def main():
    """main func"""
    pass


if __name__ == "__main__":
    main()
