from typing import Any, Generic, Hashable, Tuple, Optional, Sequence, Dict, Protocol, TypeVar, Union, overload
import random
from itertools import islice, cycle

from utils.window import window

### Exceptions

class GeomError(Exception): ...
class IDError(GeomError): ...
class ContainerError(GeomError): ...

### Types

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Vec = TypeVar('Vec', Vec2, Vec3) 
AnyVec = Union[Vec2, Vec3]

Identifier = Hashable
Collection = Dict[Identifier, Any]

# class Identified(Protocol):
#     @property
#     def id(self) -> Identifier: ...
#     def has_id(self) -> bool: ...
#     def set_id(self, id: Hashable): ...

### Classes

class Identifiable(object):
    """An object is `identifiable` if it has:
       - self.id (and self.id is immutable)
       - has_id() -> bool
       - set_id()
    """
    def __init__(self):
        self._id: Identifier = None
    def has_id(self) -> bool:
        try:
            return self._id is not None
        except:
            return False
    @property
    def id(self):
        try: 
            return self._id
        except:
            return None
    def set_id(self, id: Identifier):
        if (not hasattr(self, '_id')) or self._id is None:
            self._id = id
        else:
            raise IDError("ID cannot be changed")
class Containable(Identifiable):
    """An element is containable if it is `Identifiable` and it has a container object.
    """
    @property
    def container(self) -> 'Universe':
        try:
            return self._container 
        except:
            return None # type: ignore
    @container.setter
    def container(self, container: 'Universe'):
        if self in container:
            self._container = container
    def add_as_sibling(self, other, strict=False):
        """Adds a new object to the same container as this object. The Container will provide an id if there is not one already specified."""
        if self.container is not None:
            other.insert_into(self.container)
        elif strict:
            raise ContainerError("No container")
    def insert_into(self, container: 'Universe'):
        if self in container: # Am I already in this container?
            return
        container._put(self) # Add me to the universe, generating an ID if it is needed
        self.container = container # Make sure I know what universe I am in
        
class Element(Containable):
    ...

class Universe(object):
    def __init__(self, name: Identifier):
        self.name = name
        self._contents: Collection = {}
        self._next_id = 0
    def __repr__(self):
        return f"Universe('{self.name}': {len(self._contents)} objects)"
    def get(self, id: Identifier) -> Identifiable:
        return self._contents[id]
    def _put(self, obj: Containable):
        """Add an element to this container. If no ID is given, it will be generated. This is a private method"""
        if obj.id is not None: # Identified object
            if obj not in self: # Not already present
                self._contents[obj.id] = obj # Add it
                obj.container = self
            else: # Already there
                raise IDError(f"The id '{obj.id}' is already present in this universe")
        else: # unidentified
            newid = self.next_id
            obj.set_id(newid)
            self._contents[newid] = obj
            obj.container = self
    @property
    def next_id(self):
        "Compute a new identifier"
        while self._next_id in self._contents:
            self._next_id += 1
        return self._next_id
    def __contains__(self, other: Identifiable) -> bool:
        return other.id in self._contents

class Point(Element, Generic[Vec]):
    def __init__(self, coord: Vec, id: Identifier=None):
        self.coord: Vec = coord
        if id:
            self._id = id
        else:
            self._id = None
    @property
    def dim(self):
        return len(self.coord)
    def __repr__(self):
        return f'Point(<{self.id}>, {self.coord})'
    
    def __add__(self, other: "Point") -> "Point":
        coord: Vec = tuple([sum(a) for a in zip(self.coord, other.coord)]) # type: ignore
        return Point(coord)
    def __mul__(self, other: float) -> "Point":
        coord: Vec = tuple(coord*other for coord in self.coord) # type: ignore
        return Point(coord) 

Points = Sequence[Point]

def linterp(a: Vec, b: Vec, x: float) -> Vec:
    coord = tuple(ai+(bi-ai)*x for ai,bi in zip(a,b))
    return coord # type: ignore

class Edge(Element):
    def __init__(self, a: Point, b: Point, id: Identifier = None):
        assert a.container is b.container, "Points must be from the same universe"
        self.a = a
        self.b = b
        if id:
            self._id = id
        self.a.add_as_sibling(self)
    def mid_point(self) -> Point:
        return Point(linterp(self.a.coord, self.b.coord, 0.5))
    def insert_into(self, container: 'Universe'):
        if self in container: # Am I already in this container?
            return
        for obj in (self, self.a, self.b):
            container._put(obj) # Add me to the universe, generating an ID if it is needed
            obj.container = container # Make sure I know what universe I am in

class Poly(Element):
    def __init__(self, points: Sequence[Point]):
        self._points = [point for point in points]
    def n(self):
        return len(self._points)
    def is_self_intersecting(self) -> bool:
        ...
    def points(self, close_loop=False):
        return cycle(self._points + [self._points[0]])

class Quad(Poly):
    def __init__(self, edge_a: Point, edge_b: Point, outer: Point, inner: Point):
        assert edge_a.container is edge_b.container is outer.container is inner.container, "Points must be from the same universe"
        self.edge = Edge(edge_a, edge_b)
        self.norm_edge = Edge(inner, outer)

        

u = Universe("default")
p = Point((1,2))
print(p.container)
p.insert_into(u)

print(p in u)
print(p.container)
q = Point((2,3))
p.add_as_sibling(q)
print(q.container)

e = Edge(p,q)

# def is_ordered(points: Sequence[Point]) -> bool:
#     """Takes a sequence of Points. Returns the ordering of these points, if it exists.

#     Args:
#         points (Sequence[Point]): A sequence of point objects

#     Returns:
#         bool: 1 if the objects are in counterclockwise order
#               0 if the objects are not ordered
#               -1 if the objects are in clockwise order 
#     """
#     start_point = points[0]

#     return False

# def tri_area(A: Point, B: Point, C: Point):
#     x1,y1 = A.coord
#     x2,y2 = B.coord
#     x3,y3 = C.coord
#     return 0.5 * (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)

# def poly_area(points: Points) -> float:
#     a = 0
#     for p,q in window(points,2):
#         a += (p.coord[0]*q.coord[1]-p.coord[1]*q.coord[0])
#     return a/2

# def signed_poly_area_breakdown(points: Points) -> Sequence[float]:
#     x0,y0 = points[0].coord
#     s = []
#     for p,q in window(points[1:]):
#         s.append()
#     return s




# p = Point(1)
# u = Universe("universe")
# u.put(p)

# class q(object):
#     def __init__(self, p):
#         self.id = p

# l = q(4)
# u.put(l)

