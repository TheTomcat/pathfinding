from typing import Callable, List, Protocol, Tuple, runtime_checkable
import logging

@runtime_checkable
class DynamicObserver(Protocol):
    def handle_update(self, subject: 'Subject', *args, **kwargs): ...

@runtime_checkable
class Subject(Protocol):
    def notify_observers(self):... 
    def register_observer(self, observer:'DynamicObserver'): ...
    def deregister_observer(self, observer: 'DynamicObserver'): ...
    _observers: List[DynamicObserver]

class BaseDynamicSubject(object):
    """Implements a subject-observer pattern whereupon the modification of the subject (in this case, a point,
    results in an update command being set to all observers. If they are lazy they can flag themself as stale
    or they can immediately update. Easy. Inherit from BaseDynamicSubject to have subscribers, and for observers 
    implement an update function f(subject, *args, **kwargs) which makes appropriate changes. 

    Event messages can be sent from subject to observer via the update function. The subject keeps a record of 
    all subscribed observers, but the observers need not keep a record of what subject they are subscribed to.
    The subject is passed as the first argument of observer.update()"""
    def __init__(self):
        self._dynamic=True
        self._observers: List[DynamicObserver] = []
    def notify_observers(self, *args, **kwargs):
        for observer in self._observers:
            observer.handle_update(self, *args, **kwargs)
    def register_observer(self, observer: DynamicObserver):
        self._observers.append(observer)
    def deregister_observer(self, observer: DynamicObserver):
        self._observers.append(observer)

def internalise_arguments(func):
    """Decorator which allows internalisation of arguments to a function 
    so something like f(*args, **kwargs) -> Any is rescoped to 
    f() -> Any.

    Useful for recomputing the value of properties that have dependencies.    
    """
    def middle(*arguments):
        def inner():
            return func(*arguments)
        return inner
    return middle

class DynamicPoint(BaseDynamicSubject):
    """An example implementation of BaseDynamicSubject - a 2D coordinate with
    x and y coordinates. Access to the attributes _x and _y is managed through
    getters and setters which accordingly update the observers when a change occurs. 
    The event sent can be modified as needed. This is an example.

    Args:
        BaseDynamicSubject ([type]): [description]
    """
    def __init__(self, x: float, y: float):
        super().__init__()
        self._x = x
        self._y = y
    @property
    def x(self) -> float:
        return self._x
    @x.setter
    def x(self, value:float):
        self._x, old = value, self._x
        self.notify_observers(x=value, old=old)
    @property
    def y(self) -> float:
        return self._y
    @y.setter
    def y(self, value:float):
        self._y, old = value, self._y
        self.notify_observers(y=value, old=old)
    @property
    def point(self):
        return self._x, self._y
    @point.setter
    def point(self, value: Tuple[float, float]):
        self._x, oldx = value[0], self._x
        self._y, oldy = value[1], self._y
        self.notify_observers(point=value, old=(oldx, oldy))

if __name__ == "__main__":
    class DumbObserver(object):
        def __init__(self, p: Subject, label):
            self.label = label
            p.register_observer(self)
        def handle_update(self, subject: Subject, *args, **kwargs):
            print(f"I am {self.label} and have been notified - got {args} and {kwargs} from {subject}")

    subject = DynamicPoint(2,2)
    #observers = []
    for i in range(10):
        DumbObserver(subject, str(i))
    subject.x = 4