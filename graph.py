# graph.py
from typing import Any, Hashable, Dict, Iterable, List, Protocol, Sequence, Tuple, TypeVar, Optional, Callable, TypedDict
from collections import deque
from functools import total_ordering
import math

from utils.priority_queue import PriorityQueue, Queue

ID = Hashable

# http://yaroslavvb.com/upload/graphs2.txt ? Test graphs

class GraphError(Exception): ...
class DirectedGraphError(GraphError): ...
class WeightedGraphError(GraphError): ...
class NodeError(GraphError): ...
class EdgeError(GraphError): ...

class NodeLike(Protocol):
    _connections: Dict['NodeLike', float]
    def join(self, n: 'NodeLike', weight: Optional[float]): ...
    def neighbours(self) -> Sequence['NodeLike']: ...
    @property
    def id(self) -> Hashable: ...

@total_ordering
class Node(object):
    def __init__(self, id: ID, payload=None):
        self._id = id
        self._connections: Dict['Node', float] = dict()
        self.payload = payload
    def __repr__(self):
        return self.__class__.__qualname__ + f'(id={self.id})'
    def join(self, n: 'Node', weight=1):
        self._connections[n] = weight
    def neighbours(self): #-> Iterable['Node']:
        return iter(self._connections.items())
    @property
    def id(self):
        return self._id
    def __lt__(self, other: 'Node') -> bool: #Needed for ordering within priority queue. It's honestly probably not, but I'm lazy and this was the easiest way. 
        return self.id < other.id

class NodeData(TypedDict):
    cost: float
    parent: Optional[Node]
    depth: int

Path = List[Node]
SearchResult = Dict[Node, NodeData] 
ExtendedSearchResult = Tuple[SearchResult, NodeData]

class Graph(object):
    def __init__(self, weighted=True, directed=False):
        self._nodes: Dict[ID, Node] = dict()
        self._weighted = weighted
        self._directed = directed
    def __iter__(self):
        return iter(self._nodes.values())
    def select_node(self) -> Node:
        return next(iter(self._nodes.values()))
    def add_node(self, id: ID) -> Node:
        n = Node(id)
        self._nodes[id] = n
        return n
    def get_node(self, id: ID) -> Node:
        try: 
            return self._nodes[id] # self._nodes.get(id, None)
        except KeyError as e:
            raise NodeError from e
    def __contains__(self, id: ID) -> bool:
        return id in self._nodes
    def add_edge(self, start: ID, end: ID, weight=0, weight_backwards=None):
        if weight_backwards is not None and not self._directed:
            raise DirectedGraphError("This is not a directed graph")
        if weight != 0 and not self._weighted:
            raise WeightedGraphError("This is not a weighted graph")
        if weight_backwards is None:
            weight_backwards = weight
        if start not in self:
            self.add_node(start)
        if end not in self:
            self.add_node(end)
        self._nodes[start].join(self._nodes[end],weight)
        self._nodes[end].join(self._nodes[start],weight_backwards)

    @classmethod
    def from_edge_list(cls, edges: Sequence[Sequence[ID]]) -> 'Graph':
        g = cls()
        node_index: Dict[ID,Node] = {}
        for edge in edges:
            if len(edge) != 2:
                raise EdgeError(f"Edge supplied with > 2 nodes: {edge}")
            node_a_id, node_b_id = edge
            if node_a_id not in node_index:
                node_index[node_a_id] = g.add_node(node_a_id)
            if node_b_id not in node_index:
                node_index[node_b_id] = g.add_node(node_b_id)
            g.add_edge(node_a_id, node_b_id)
        return g

class Grid(Graph):
    """Construct a graph in a grid layout, omitting nodes in the list `walls`. Node ids are given by their integer coordinates as a 2-tuple.

    Args:
        width (int): The width of the grid
        height (int): The height of the grid
        walls (List[Tuple[int,int]]): A list of 2-tuples corresponding to coordinates where walls are present
        link_8 (bool, optional): What directions make up neighbours? if True, use 8 directions (N,NE,E,SE,S,SW,W,NW). Otherwise use NSEW. Defaults to True.

    Returns:
        Graph: A graph representation of the above. 
    """
    def __init__(self, width: int, height: int, walls: List[Tuple[int,int]], link_8=True):
        super(Grid, self).__init__(weighted=True, directed=False)
        self.width = width
        self.height = height
        self.walls = walls
        self.from_grid(width, height, walls, link_8)
    def from_grid(self, width: int, height: int, walls: List[Tuple[int,int]], link_8=True):
        for y in range(height):
            for x in range(width):
                if (x,y) in walls:
                    continue
                n = self.add_node((x,y))
                for dx,dy,d in [(0,1,1),(1,1,1.414),(1,0,1),(1,-1,1.414),(0,-1,1),(-1,-1,1.414),(-1,0,1),(-1,1,1.414)]:
                    nx,ny = x+dx, y+dy
                    if not link_8 and d!=1: continue
                    if not (0<=nx<width and 0<=ny<height): continue
                    if (nx,ny) in walls: continue
                    try:
                        neighbour = self.get_node((nx,ny))
                    except NodeError as e:
                        neighbour = self.add_node((nx,ny))
                    self.add_edge((x,y),(nx,ny),d)
    def ascii_print(self, path=None, mapping: Dict[Tuple[int,int],str]=None):
        if mapping is None:
            mapping = {}
        print("___" * self.width + '__')
        for y in range(self.height):
            print('|', end="")
            for x in range(self.width):
                r = ' . '
                try:
                    node = self.get_node((x,y))
                except NodeError as e: # No such node exists
                    r = '###'
                if (x,y) in mapping:
                    r = f' {mapping[(x,y)][0]} '
                print(r, end="")
            print('|')
        print("___" * self.width + '__')

    @staticmethod 
    def l2_norm(a: Node, b: Node) -> float:
        x1,y1 = a.id
        x2,y2 = b.id
        return math.sqrt((x2-x1)**2+(y2-y1)**2)
    @staticmethod
    def l1_norm(a: Node, b: Node) -> float:
        x1,y1 = a.id
        x2,y2 = b.id
        return abs(x1-x2)+abs(y1-y2)
    @staticmethod
    def grid_dist(a: Node, b: Node) -> float:
        x1,y1 = a.id
        x2,y2 = b.id
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        low = min(dx,dy)
        dd = abs(dy-dx)
        return low*1.414 + dd

def _search(graph: Graph, start: Node, end: Callable[[Node], bool]=None, depth_first=False) -> SearchResult: #Dict[Node, Optional[Node]]:
    """Perform a breadth- or depth-first search of the graph, starting at node `start`. 
    Optionally, end the search early when the end condition is satisfied. 
    For a bredth-first search, depth_first=False. Otherwise a depth_first search is performed. 


    Args:
        graph (Graph): The graph to search
        start (Node): The starting node
        end (f(Node)-> Bool, optional): A function which, when true, will halt the search early. Defaults to f(x)->False.
        depth_first (bool, optional): Perform a depth_first search. Defaults to False.

    Returns:
        Dict[Node, Optional[Node]]: A visited_from dictionary of (key,val) pairs where d[node_a]=node_b means that node_a arrived via node_b. None represents the starting node.
    """
    if end is None:
        end = lambda x: False
    queue: Queue[Node] = Queue()
    enqueue = queue.put if depth_first else queue.putleft
    enqueue(start)
    visited_from: SearchResult = {}#Dict[Node, Optional[Node]] = {}
    visited_from[start] = {'parent':None, 'cost':0, 'depth':0}
    print(f'Starting {"depth" if depth_first else "bredth"}-first search...')
    while not queue.empty():
        current = queue.get()
        print(current.id, end=" ")
        for neighbour, weight in current.neighbours():
            if neighbour not in visited_from:
                cweight = visited_from[current]['cost']
                cdepth = visited_from[current]['depth']
                enqueue(neighbour)
                visited_from[neighbour] = {'parent':current, 'cost': cweight+weight, 'depth':cdepth+1}
            if end(neighbour):
                print(" - End condition satisfied, halting!")
                return visited_from
    return visited_from

def BFS(graph: Graph, start: Node, end: Callable[[Node], bool]=None) -> SearchResult: #Dict[Node, Optional[Node]]:
    return _search(graph, start, end=end, depth_first=False)
def DFS(graph: Graph, start: Node, end: Callable[[Node], bool]=None) -> SearchResult: #Dict[Node, Optional[Node]]:
    return _search(graph, start, end=end, depth_first=True)

# def dijkstra_search(graph: Graph, start: Node, end: Node) -> ExtendedSearchResult:
#     queue = PriorityQueue()
#     queue.put(start, 0)
#     visited_from: SearchResult = {start: {'parent':None,'cost':0,'depth':0}}
#     while not queue.is_empty():
#         current: Node = queue.get()
#         if current == end: break
#         for neighbour, weight in current.neighbours():
#             new_cost = visited_from[current]['cost'] + weight
#             new_depth = visited_from[current]['depth'] + 1
#             if neighbour not in visited_from or new_cost < visited_from[neighbour]['cost']:
#                 visited_from[neighbour] = {'parent':current, 'cost':new_cost, 'depth':new_depth}
#                 priority = new_cost
#                 queue.put(neighbour, priority)
#     return visited_from, visited_from[end]

def A_star(graph: Graph, start: Node, end: Node, heuristic: Optional[Callable[[Node,Node],float]]=None) -> ExtendedSearchResult:
    """Priority-search. Perform an A-star or Dijkstra search on the graph. If heuristic is not provided, will perform standard
    Dijkstra search. If heuristic is provided, perform A-star.

    Args:
        graph (Graph): The graph object
        start (Node): The starting node
        end (Node): The ending node
        heuristic (f(Node,Node) -> float, optional): A function f(Node,Node) -> float providing a heuristic cost. If no function is provided, perform Dijkstra search. Defaults to None.

    Returns:
        ExtendedSearchResult: Dict[Node: {'parent':Node, 'weight':float, 'depth':int}], Dict[end]
    """
    queue = PriorityQueue()
    queue.put(start, 0)
    visited_from: SearchResult = {start: {'parent':None,'cost':0,'depth':0}}
    while not queue.is_empty():
        current: Node = queue.get()
        if current == end:
            break
        for neighbour, weight in current.neighbours():
            new_cost = visited_from[current]['cost'] + weight
            new_depth = visited_from[current]['depth'] + 1
            if neighbour not in visited_from or new_cost < visited_from[neighbour]['cost']:
                visited_from[neighbour] = {'parent':current, 'cost':new_cost, 'depth':new_depth}
                if heuristic is None: # Dijkstra search
                    priority = new_cost
                else: # A-star search
                    priority = new_cost + heuristic(neighbour, end)
                queue.put(neighbour, priority)
    return visited_from, visited_from[end]

def construct_path(visited_from: SearchResult, from_node: Node) -> Path:
    """Return a path from `from_node` to the root of the spanning tree.

    Args:
        visited_from (Dict[Node, Optional[Node]]): The output of DFS or BFS.
        from_node (Node): [description]

    Returns:
        Path: [description]
    """
    path: List[Node] = []
    current_node = from_node
    while current_node != None:
        path.append(current_node)
        parent = visited_from[current_node]
        if parent['parent'] is None:
            path.reverse()
            return path
        current_node = parent['parent']
    path.reverse()
    return path


edge_list = [('a','b'),('b','c'),('c','d'),('d','l'),('d','m'),('c','f'),('f','n'),
             ('f','q'),('b','e'),('e','f'),('e','h'),('h','q'),('b','o'),('o','p'),
             ('p','t'),('t','r'),('r','s'),('s','o'),('o','h'),('a','i'),('i','g'),('i','k')]

G = Graph.from_edge_list(edge_list)
a = G.get_node('a')
h = G.get_node('h')

full_bfs = BFS(G,a)
full_dfs = DFS(G,a)
end = lambda x: x.id=='h'
bfs = BFS(G,a,end=end)
dfs = DFS(G,a,end=end)

walls = [(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8),(9,8),
         (0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(6,6),(7,6),(8,6),
         (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),
         (3,1),(3,2),(3,3)]#((1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
g = Grid(10,10, walls=walls)
A_star_result, end_data = A_star(g, g.get_node((1,9)), g.get_node((8,3)),Grid.grid_dist)
path = {node.id: "@" for node in construct_path(A_star_result, g.get_node((8,3)))}
path.update({(1,9):'S', (8,3):'F'})
g.ascii_print(mapping=path)
# g = Graph()
# a = g.add_node('a')
# b = g.add_node('b')
# c = g.add_node('c')
# d = g.add_node('d')
# e = g.add_node('e')
# f = g.add_node('f')
# h = g.add_node('h')

# g.add_edge('a','b',7)
# g.add_edge('a', 'c', 9)
# g.add_edge('a', 'f', 14)
# g.add_edge('b', 'c', 10)
# g.add_edge('b', 'd', 15)
# g.add_edge('c', 'd', 11)
# g.add_edge('c', 'f', 2)
# g.add_edge('d', 'e', 6)
# g.add_edge('e', 'f', 9)

# search(g,g.get_node('a'))
