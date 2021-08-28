import math
from typing import List

from dynamic_element import BaseDynamicSubject, DynamicObserver, LazyObserver, Subject
from vec2 import Vec2
from utils import window

class Poly(BaseDynamicSubject, LazyObserver):
    def __init__(self, *points: Vec2):
        self._points = points
        self._fresh = False
        for point in points:
            point.register_observer(self)
        super().__init__()
    def update(self, subject: Subject, *args, **kwargs):
        if not self._fresh:
            self.compute_area()
    def area(self):
        self.update()
        return self._area
    def compute_area(self):
        tot = 0
        tot += sum(p.x*q.y - q.x*p.y for p,q in window(self._points))
        tot += (self._points[-1].x*self._points[0].y)
        tot -= (self._points[0].x*self._points[-1].y)
        self._area = 0.5*tot

if __name__ == "__main__":
    a = Vec2(1,1)
    b = Vec2(0,0)
    c = Vec2(1,0)
    p = Poly(a,b,c)
    print(p.area())
    c.x=2
    print(p.area())