import math
from typing import Any, Callable, List, Tuple, Union
from dynamic_element import DynamicPoint, DynamicObserver, Subject

Number = Union[float, int]

def add(a: 'Vec2', b: 'Vec2') -> Callable:
    """Function factory returning a function that computes the sum of two vectors"""
    return lambda : (a.x+b.x, a.y+b.y)

def mul(a: 'Vec2', b: Number) -> Callable:
    """Function factory returning a function that computes a scaled vector"""
    return lambda : (a.x*b, a.y*b)

def unit(a: 'Vec2'):
    """Function factory creating a unit vector"""
    mag = a.mag()
    return lambda : (a.x/mag, a.y/mag)

class Vec2(DynamicPoint):
    def __init__(self, x: float, y: float, lazy=False):
        super().__init__(x, y)
        self._lazy = lazy
        self._stack: List[Tuple[Callable, List[Any]]] = []

    def update(self, subject: 'Subject', *args, **kwargs):
        if self._lazy:
            self._fresh = False
        else:
            self.recompute()
    
    def recompute(self):
        for action, func_factory in self._stack:
            action(*func_factory)

    def update_self(self, func: Callable[[],Tuple[float,float]]):
        """Update my own values using the function func() -> (x,y)"""
        vals = func()
        self._x, oldx = vals[0], self._x
        self._y, oldy = vals[1], self._y
        self.notify_observers(point=vals, old=(oldx, oldy))

    def mag(self):
        return math.sqrt(self._x*self._x + self._y*self._y)

    def __add__(self, other: 'Vec2') -> 'Vec2':
        v = Vec2(self.x+other.x, self.y+other.y, lazy=False)
        self.register_observer(v)
        other.register_observer(v)
        v._stack.append((v.update_self,[add(self, other)]))
        return v
    
    def __mul__(self, other: Number) -> 'Vec2':
        v = Vec2(self.x*other, self.y*other, lazy=False)
        self.register_observer(v)
        v._stack.append((v.update_self, [mul(self, other)]))
        return v

    def __repr__(self):
        return f'Vec2({self.x},{self.y})'


if __name__ == "__main__":
    a = Vec2(1,1)
    b = Vec2(2,2)
    print(a._observers)
    c = a+b
    print(c._observers)
    print(c)
    a.x=2
    print(c)
    
    d = a*2
    print(a, d)    

    a.y = 4
    print(a, d)