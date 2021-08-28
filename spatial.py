from typing import Any, Callable, Hashable, Iterable, List, Protocol, Tuple

ID = Hashable

class PointLike(Protocol):
    x: float
    y: float
    point: Tuple[float, float]

class SpatialElement(Protocol):
    id: ID
    def bounding_box(self) -> Tuple[PointLike, PointLike]: ...
    @property
    def points(self) -> Iterable[PointLike]: ...

class Environment(Protocol):
    _points: List[PointLike]
    _elements: List[SpatialElement]
    def generate_id(self) -> ID: ...
    def add_point(self, point: PointLike): ...
    def add_points(self, points: List[PointLike]): ...
    def add_element(self, element: SpatialElement): ...
    def add_elements(self, elements: List[SpatialElement]): ...
    def __contains__(self, x: Any) -> bool: ...
