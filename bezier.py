import numpy as np
from typing import Any, List, Optional, Protocol, Tuple

class PointLike(Protocol):
    x: int
    y: int
    @property
    def point(self) -> Tuple[float]: ...

class Vec2():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    @property
    def point(self):
        return self.x, self.y
    def __add__(self, other: Any) -> 'Vec2':
        return Vec2(self.x+other.x, self.y+other.y)
    def __mul__(self, other: Any) -> 'Vec2':
        if isinstance(other, (int,float)):
            return Vec2(other*self.x, other*self.y)
        else:
            raise ValueError(f"Cannot multiple {type(other)} and Vec2")

class Bezier(object):
    pass

class BezierLin(Bezier):
    pass

class BezierQuad(Bezier):
    pass

b3p0t = lambda t:  -t*t*t + 3*t*t - 3*t + 1
b3p1t = lambda t: 3*t*t*t - 6*t*t + 3*t
b3p2t = lambda t:-3*t*t*t + 3*t*t
b3p3t = lambda t: t*t*t

b3p0tp = lambda t: -3*t*t + 6*t - 3
b3p1tp = lambda t:  9*t*t -12*t + 3
b3p2tp = lambda t: -9*t*t + 6*t 
b3p3tp = lambda t:  3*t*t 

b3p0tpp = lambda t: -6*t + 6
b3p1tpp = lambda t: 18*t - 12
b3p2tpp = lambda t:-18*t + 6
b3p3tpp = lambda t:  6*t

chk = lambda points: sum([(point.x+point.y)**(i+1) for i, point in enumerate(points)])

class BezierCubic(Bezier):
    def __init__(self, P0: PointLike, P1: PointLike, P2: PointLike, P3: PointLike, dynamic=True):
        self.points: List[PointLike] = [P0, P1, P2, P3]
        self.dynamic: bool = dynamic
        if dynamic:
            self._internal_points = [P0.x, P1.x, P2.x, P3.x, P0.y, P1.y, P2.y, P3.y]
        self._velocity: Optional[BezierQuad] = None
        self._accel: Optional[BezierLin] = None
        self._curvature = None
        self._bounding_box = None
    
    def _current(self) -> bool:
        if not self.dynamic:
            return True
        return all((self.points[0].point == self._internal_points[0], self._internal_points[4], 
                    self.points[1].point == self._internal_points[1], self._internal_points[5],
                    self.points[2].point == self._internal_points[2], self._internal_points[6],
                    self.points[3].point == self._internal_points[3], self._internal_points[7]))
    def __call__(self, t: float) -> PointLike:
        return Vec2(t,t)
    def bound(self):
        pass
    def d1(self):
        pass
    def d2(self):
        pass
    def k(self):
        pass


class BezierSpline(object):
    pass
