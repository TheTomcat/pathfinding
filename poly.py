import math
from typing import Dict, List

from dynamic.dynamic_element import BaseDynamicSubject, Subject
from dynamic.cacheable_property import CacheableMixin, cacheable_property
from vec2 import Vec2
from utils.window import window

class Poly(CacheableMixin):
    def __init__(self, *points: Vec2):
        self._points = [point for point in points]
        for point in points:
            point.register_observer(self)

    def handle_update(self, subject: Subject, *args, **kwargs):
        self.mark_as_stale()

    @cacheable_property
    def area(self):
        """The area calculation"""
        tot = 0
        tot += sum(p.x*q.y - q.x*p.y for p,q in window(self._points))
        tot += (self._points[-1].x*self._points[0].y)
        tot -= (self._points[0].x*self._points[-1].y)
        return math.fabs(0.5*tot)

    @cacheable_property 
    def bounding_box(self):
        xmin=ymin=math.inf
        xmax=ymax=-math.inf
        for point in self._points:
            x,y = point.point
            xmin = min(x,xmin)
            xmax = max(x,xmax)
            ymin = min(y,ymin)
            ymax = max(y,ymax)
        return (xmin,ymin),(xmax,ymax) 

    def add_vertex(self, point: Vec2):
        self._points.append(point)
        point.register_observer(self)
        self.mark_as_stale()

    def remove_vertex(self, point:Vec2):
        self._points.remove(point)
        point.deregister_observer(self)
        self.mark_as_stale()

if __name__ == "__main__":
    a = Vec2(0,0)
    b = Vec2(0,1)
    c = Vec2(1,1)
    d = Vec2(1,0)
    p = Poly(a,b,c)
    print(p.area())
    p.add_vertex(d)
    print(p.area())
    c.x=2
    print(p.area())
    #a=Vec2(0,0) ; b=Vec2(0,1) ; c=Vec2(1,1) ; d=Vec2(1,0) ; p=Poly(a,b,c)