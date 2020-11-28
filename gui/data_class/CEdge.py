# coding: utf-8
""""""

from typing import Callable, Dict, List, Tuple

import numpy as np

from .CPoint import CPoint

import logging


class CEdge(object):
    """エッジというか直線のクラス、線分である機能はまだない"""

    def __init__(self, p0: CPoint, p1: CPoint) -> None:
        self.set_points(p0, p1)

    def set_points(self, p0: CPoint, p1: CPoint) -> None:
        self.set_p0(p0)
        self.set_p1(p1)

    def set_p0(self, p0: CPoint) -> None:
        self._p0 = p0

    def set_p1(self, p1: CPoint) -> None:
        self._p1 = p1

    @property
    def p0(self,) -> CPoint:
        return self._p0

    @property
    def x0(self,) -> CPoint:
        return self.p0.x

    @property
    def y0(self,) -> CPoint:
        return self.p0.y

    @property
    def p1(self,) -> CPoint:
        return self._p1

    @property
    def x1(self,) -> CPoint:
        return self.p1.x

    @property
    def y1(self,) -> CPoint:
        return self.p1.y

    @property
    def a(self,) -> float:
        return self.y1 - self.y0

    @property
    def b(self,) -> float:
        return self.x0 - self.x1

    @property
    def c(self,) -> float:
        return -1*self.b*self.y0 - 1*self.a*self.x0

    @property
    def length(self) -> float:
        return self.p0.dist(self.x1, self.y1)

    def __str__(self):
        res = [
            self.__class__.__name__,
            str(self.p0),
            str(self.p1),
        ]
        return ":".join(res)

    def __iter__(self):
        raise NotImplementedError()

    def dist(self, x, y):
        return np.abs(self.a*x+self.b*y+self.c)/(self.a**2+self.b**2)**0.5

    def on_edge(self, x, y):
        logger = logging.getLogger(__name__)

        base = CPoint(self.p1.x - self.p0.x, self.p1.y-self.p0.y)
        tgt = CPoint(x - self.p0.x, y-self.p0.y)
        ret = tgt.dot(base)
        logger.info(self)
        logger.info(self.length)
        proj_len = ret / self.length

        if 0 <= proj_len and proj_len <= self.length:
            return True
        else:
            return False


def main():
    """main func"""
    pointA = CPoint(0, 0)
    pointB = CPoint(2, 2)
    pointC = CPoint(3, 0)

    edge = CEdge(pointA, pointB)
    print(pointA)
    print(pointB)
    print(pointC)
    print(edge)
    print(edge.dist(pointC.x, pointC.y))

    print(edge.on_edge(pointC.x, pointC.y))


if __name__ == "__main__":
    main()
