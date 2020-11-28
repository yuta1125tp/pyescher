# coding: utf-8
""""""
#
from typing import Tuple


class CPoint(object):
    def __init__(self, x: float, y: float) -> None:
        self.setxy(x, y)
        self.dims = 2

    def __str__(self) -> str:
        res = self.__class__.__name__
        res += ' (x, y) = ({}, {})'.format(self._x, self._y)
        return res

    def __len__(self) -> int:
        return self.dims

    def dist(self, x: float, y: float) -> float:
        return ((self._x - x)**2 + (self._y - y)**2)**0.5

    def setxy(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def tolist(self) -> Tuple[float, float]:
        return [self._x, self._y]

    # def dot(self, point: CPoint) -> float:
    def dot(self, point: 'CPoint') -> float:
        """in Python<3.7 we cannot attnotate class using itself.  
        [ref](https://stackoverflow.com/a/33533514)  
        once it move to 3.7 we can fix this behavior.  
        ```python
        from __future__ import annotations
        ...
        def dot(self, point: CPoint) -> float:
            pass

        ```
        """
        return self.x*point.x + self.y*point.y


def main():
    """main func"""
    pass


if __name__ == "__main__":
    main()
