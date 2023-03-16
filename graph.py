# graph.py
from typing import (Generator, Hashable, Dict, ItemsView, Iterable, 
        List, Protocol, Sequence, Set, Tuple, TypeVar, Optional, 
        Callable, TypedDict, Union)
from collections import deque
from functools import total_ordering
import math

import disjoint_set

from utils.priority_queue import PriorityQueue, Queue
from utils.documented_by import is_documented_by

ID = Hashable

# http://yaroslavvb.com/upload/graphs2.txt ? Test graphs

class GraphError(Exception): ...
class DirectedGraphError(GraphError): ...
class WeightedGraphError(GraphError): ...
class NodeError(GraphError): ...
class EdgeError(GraphError): ...

@total_ordering
class Node(object):
    def __init__(self, id: ID, payload=None):
        self._id = id
        self._connections: Dict['Node', float] = dict()
        self.payload = payload
    def __repr__(self):
        return self.__class__.__qualname__ + f'(id={self.id})'
    def join(self, n: 'Node', weight=1):
        """Add an edge from this node to `n` with optional `weight`"""
        self._connections[n] = weight
    def neighbours(self) -> Generator[Tuple['Node',float], None, None]: #-> Iterable['Node']:
        """Iterate over the neighbouring nodes -> [(neighbour:Node, weight:float), ...]"""
        yield from self._connections.items()
    @property
    def id(self):
        return self._id
    def __lt__(self, other: 'Node') -> bool: 
        # Hack for ordering within priority queue when nodes have the same priority. It's honestly probably not, but I'm lazy and this was the easiest way. 
        # Obviously can't be used with ids of mixed types. One day I'll fix this, but it isn't a problem for me right now, so it's staying.
        return self.id < other.id 

    def get_weight(self, other: 'Node'):
        try:
            return self._connections[other]
        except KeyError:
            return None
    def __len__(self):
        return len(self._connections)

class NodeData(TypedDict):
    cost: float
    parent: Optional[Node]
    depth: int


Weight = Union[float, int]
Edge = Union[Tuple[ID,ID],Tuple[ID,ID,Weight]]

Path = List[Node]
SearchResult = Dict[Node, NodeData] 
ExtendedSearchResult = Tuple[SearchResult, Node]
EndCondition = Callable[[Node],bool] 
Heuristic = Callable[[Node,Node],float]#Union[...,Callable[[Node,Node,'Graph'],float]]
class NodeLike(Protocol):
    _connections: Dict['NodeLike', float]
    def join(self, n: 'NodeLike', weight: Optional[float]): ...
    def neighbours(self) -> Sequence['NodeLike']: ...
    @property
    def id(self) -> Hashable: ...


class Graph(object):
    def __init__(self, weighted=True, directed=False):
        self._nodes: Dict[ID, Node] = dict()
        self._weighted = weighted
        self._directed = directed
    @property
    def weighted(self):
        return self._weighted
    @property
    def directed(self):
        return self._directed
    def __iter__(self):
        return iter(self._nodes.values())
    def nodes(self) -> Generator[Tuple[ID,Node], None, None]:
        yield from self._nodes.items()
    def nodes_by_id(self) -> Generator[ID, None, None]:
        print("Depreciated call nodes_by_id, instead use for id, node in nodes()")
        yield from self._nodes.keys()
    def nodes_by_node(self) -> Generator[Node, None, None]:
        print("Depreciated call nodes_by_node, instead use for id, node in nodes()")
        yield from self._nodes.values()
    def edges(self) -> Generator[Tuple[Node, Node, float], None, None]:
        for _, node in self.nodes():
            for neighbour, weight in node.neighbours():
                yield (node, neighbour, weight)
    # def select_node(self) -> Node:
    #     """Select a random node from this graph"""
    #     return next(iter(self._nodes.values()))
    def add_node(self, id: ID) -> Node:
        "Create a node by ID and add it to the graph. Return the new node. If it already exists, return that node"
        if id in self._nodes:
            return self._nodes[id]
        n = Node(id)
        self._nodes[id] = n
        return n
    def get_node(self, id: ID) -> Node:
        "Fetch a node by ID from the graph."
        try: 
            return self._nodes[id] # self._nodes.get(id, None)
        except KeyError as e:
            raise NodeError from e
    def __contains__(self, id: Union[ID, Node]) -> bool:
        "Does a node exist in this graph? Search by Node object or ID"
        if isinstance(id, Node):
            return id.id in self._nodes
        else:
            return id in self._nodes
    def __len__(self):
        return len(self._nodes)
    def add_edge(self, start: ID, end: ID, weight=1, weight_backwards=None):
        """Add an edge to the graph from start -> end.

        Examples:
        g = Graph(weighted=True, directed=True) ; g.add_node('a') ; g.add_node('b')
        g.add_edge('a','b',1) # Adds an edge from a->b with weight 1
        g.add_edge('a','b',1,2) # Adds an edge from a->b with weight 1 AND b->a with weight 2

        h = Graph(weighted=False, directed=True) ; h.add_node('a') ; h.add_node('b')
        h.add_edge('a','b') # No weight needed, adds an unweighted edge a->b
        h.add_edge('b','a') # Again, no weight needed, adds an unweighted edge b->a

        i = Graph(weighted=True, directed=False) ; i.add_node('a') ; i.add_node('b')
        i.add_edge('a','b',3) # Adds an edge from a->b with weight 3 and from b->a with weight 3

        j = Graph(weighted=False, directed=False) ; j.add_node('a') ; j.add_node('b')
        j.add_edge('a','b') # Adds an unweighted edge a->b AND an unweight edge from b->a

        Args:
            start (ID): The ID of the starting node
            end (ID): The ID of the ending node
            weight (float, optional): The weight of the forward direction edge. Raises an error if the graph is unweighted. Defaults to 0.
            weight_backwards (float, optional): The weight of the reverse edge. Defaults to None.

        Raises:
            DirectedGraphError: [description]
            WeightedGraphError: [description]
        """
        # You have given a backwards weight for an undirected graph. Error
        if weight_backwards is not None and not self._directed: 
            raise DirectedGraphError("This is not a directed graph. You cannot specify a backwards weight for an undirected graph.")
        # You have given a weight for an unweighted graph. Error
        if ((weight != 1) or (weight_backwards is not None)) and not self._weighted: 
            raise WeightedGraphError(f"This is not a weighted graph. You cannot specify a forwards or a backwards weight: weight:{weight}, backwards_weight:{weight_backwards}")
        if start not in self:
            self.add_node(start)
        if end not in self:
            self.add_node(end)
        # Add the edge from start -> end with weight 
        self._nodes[start].join(self._nodes[end],weight) 
        # If we are an undirected graph, add the edge from end->start with weight
        if not self._directed:
            self._nodes[end].join(self._nodes[start],weight)
        # Special case: We are a directed graph, and the user has specified a backwards_weight. Add this edge.
        if self._directed and weight_backwards is not None:
            self._nodes[end].join(self._nodes[start],weight_backwards)
        
    @classmethod
    def from_edge_list(cls, edges: Sequence[Sequence[ID]], weighted: bool=False, directed:bool=False) -> 'Graph':
        g = cls(weighted=weighted, directed=directed)
        # Keep track of the nodes we've mapped so far. 
        # I guess you could also peek inside the graph object? 
        # if g.get_node(node_a_id) is None
        node_index: Dict[ID,Node] = {} 
        if weighted:
            expected_length = 3 # (node, node, weight)
        else:
            expected_length = 2
        for edge in edges:
            try:
                if weighted:
                    node_a_id, node_b_id, weight = edge
                else:
                    node_a_id, node_b_id = edge
            except ValueError as e:
                raise EdgeError(f"Edge supplied with {len(edge)} nodes, expected {expected_length}: Edge={edge}") from e
            if node_a_id not in node_index:
                node_index[node_a_id] = g.add_node(node_a_id)
            if node_b_id not in node_index:
                node_index[node_b_id] = g.add_node(node_b_id)
            if not weighted:
                g.add_edge(node_a_id, node_b_id)
                if not directed:
                    g.add_edge(node_b_id, node_a_id)
            elif weighted:
                g.add_edge(node_a_id, node_b_id, weight)
                if not directed:
                    g.add_edge(node_b_id, node_a_id, weight)
        return g
    
    @classmethod
    def from_adjacency_list(cls, adj_list: Dict[str,Sequence], weighted: bool=True, directed:bool=True) -> 'Graph':
        g = cls(weighted=weighted, directed=directed)
        for node, adjacents in adj_list.items():
            g.add_node(node)
            if isinstance(adjacents, dict):
                loop_over: Union[ItemsView,list] = adjacents.items()
            elif isinstance(adjacents, list):
                loop_over = adjacents
            for e in loop_over:
                if weighted:
                    neighbour, weight = e
                else:
                    neighbour = e
                g.add_node(neighbour)
                if weighted:
                    g.add_edge(node, neighbour, weight)
                else:
                    g.add_edge(node, neighbour)
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
                    if (nx,ny) not in self:
                        self.add_node((nx,ny))
                    # try:
                    #     neighbour = self.get_node((nx,ny))
                    # except NodeError as e:
                    #     neighbour = self.add_node((nx,ny))
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

def _search(graph: Graph, start: Node, end: EndCondition=None, depth_first=False) -> Union[SearchResult,ExtendedSearchResult]: #Dict[Node, Optional[Node]]:
    """Perform a breadth- or depth-first search of the graph, starting at node `start`. 
    Optionally, end the search early when the end condition is satisfied. 
    For a bredth-first search, depth_first=False. Otherwise a depth_first search is performed. 

    Args:
        graph (Graph): The graph to search
        start (Node): The starting node
        end (f(Node)-> Bool, optional): A function which, when true, will halt the search early. Defaults to f(x)->False.
        depth_first (bool, optional): Perform a depth_first search. Defaults to False.

    Returns:
        Dict[Node: {'parent':Node, 'weight':float, 'depth':int}]
            A visited_from dictionary of (key,val) pairs where d[node_a]=NodeData(node_b) means that node_a arrived via node_b. `None` represents the starting node.
        end: The end node, if `end` is specified
    """
    if end is None:
        end = lambda x: False
    queue: Queue[Node] = Queue()
    enqueue = queue.put if depth_first else queue.putleft
    enqueue(start)
    visited_from: SearchResult = {}#Dict[Node, Optional[Node]] = {}
    visited_from[start] = {'parent':None, 'cost':0, 'depth':0}
    print(f'Starting {"depth" if depth_first else "bredth"}-first search...')
    while not queue.is_empty():
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
                return visited_from, neighbour
    return visited_from

@is_documented_by(_search)
def BFS(graph: Graph, start: Node, end: Callable[[Node], bool]=None) -> Union[ExtendedSearchResult,SearchResult]: #Dict[Node, Optional[Node]]:
    return _search(graph, start, end=end, depth_first=False)
@is_documented_by(_search)
def DFS(graph: Graph, start: Node, end: Callable[[Node], bool]=None) -> Union[ExtendedSearchResult,SearchResult]: #Dict[Node, Optional[Node]]:
    return _search(graph, start, end=end, depth_first=True)

def A_star(graph: Graph, start: Node, end: Node, heuristic: Optional[Heuristic]=None) -> ExtendedSearchResult:
    """Priority-search. Perform an A-star or Dijkstra search on the graph. If heuristic is not provided, will perform standard
    Dijkstra search. If heuristic is provided, perform A-star.

    A heuristic is a function of the form f(Node1, Node2)->float

    Args:
        graph (Graph): The graph object
        start (Node): The starting node
        end (Node): The ending node
        heuristic (f(Node,Node) -> float, optional): A function f(Node,Node) -> float providing a heuristic cost. If no function is provided, perform Dijkstra search. Defaults to None.

    Returns:
        ExtendedSearchResult: Dict[Node: {'parent':Node, 'weight':float, 'depth':int}]
        End: The end node
    """
    queue = PriorityQueue[Node]()
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
    return visited_from, end

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

def floyd_warshall(g: Graph) -> Dict[ID, Dict[ID,float]]:
    # construct dist matrix V x V initialised to infinity
    dist = {n:{m:math.inf for m,_ in g.nodes()} for n,_ in g.nodes()}
    for n1,n2,w in g.edges():
        dist[n1.id][n2.id] = w
    for n, _ in g.nodes():
        dist[n][n] = 0
    for k,_ in g.nodes():
        for i, _ in g.nodes():
            for j, _ in g.nodes():
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def floyd_warshall_with_path(g: Graph):
    # construct dist matrix V x V initialised to infinity
    dist = {n:{m:math.inf for m,_ in g.nodes()} for n,_ in g.nodes()}
    next: Dict[ID, Dict[ID, Optional[Node]]] = {n:{m:None for m in g.nodes_by_id()} for n in g.nodes_by_id()}
    for n1,n2,w in g.edges():
        dist[n1.id][n2.id] = w
        next[n1.id][n2.id] = n2
    for n in g.nodes_by_node():
        dist[n.id][n.id] = 0
        next[n.id][n.id] = n
    for k in g.nodes_by_id():
        for i in g.nodes_by_id():
            for j in g.nodes_by_id():
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next[i][j] = next[i][k]

    return dist, next

def prim_MST(g: Graph, start: Node) -> Tuple[Graph, float]:
    if g.directed:
        raise GraphError("Cannot compute MST on a directed graph")
    mst = Graph(directed=False, weighted=g.weighted)
    #visited: Set[ID] = set(start.id)
    cost: float = 0
    edges = PriorityQueue[Tuple[ID, ID]]()
    for to, weight in start.neighbours():
        edges.put((start.id, to.id), weight)
        # https://bradfieldcs.com/algos/graphs/prims-spanning-tree-algorithm/
    mst.add_node(start.id)
    while not edges.is_empty():
        weight, (a_id, b_id) = edges.get_with_priority()
        if b_id not in mst:
            mst.add_node(b_id)
            mst.add_edge(a_id, b_id, weight)
            cost += weight
            for neighbour, weight in g.get_node(b_id).neighbours():
                if neighbour.id not in mst:
                    edges.put((b_id, neighbour.id), weight)
    return mst, cost

def kruskal_MST(g: Graph, start: Node) -> Tuple[Graph, float]:
    if g._directed:
        raise GraphError("Cannot compute MST on a directed graph")
    mst = Graph(weighted=g.weighted, directed=False)
    dsu: disjoint_set.DisjointSet[Node] = disjoint_set.DisjointSet()
    cost: float = 0
    # Create edge list
    edge_list = [(w,a,b) for (a,b,w) in g.edges()]
    edge_list.sort()
    for w,a,b in edge_list:
        if not dsu.connected(a,b):
            cost += w
            mst.add_node(a.id)
            mst.add_node(b.id)
            mst.add_edge(a.id, b.id, w)
            dsu.union(a,b)
    return mst, cost

        
# if __name__ == "__main__":
#     ...
edge_list = [('a','b'),('b','c'),('c','d'),('d','l'),('d','m'),('c','f'),('f','n'),
            ('f','q'),('b','e'),('e','f'),('e','h'),('h','q'),('b','o'),('o','p'),
            ('p','t'),('t','r'),('r','s'),('s','o'),('o','h'),('a','i'),('i','g'),('i','k')]

G = Graph.from_edge_list(edge_list, weighted=False, directed=False)
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

AL = {
    'A': {'B': 2, 'C': 3},
    'B': {'A': 2, 'C': 1, 'D': 1, 'E': 4},
    'C': {'A': 3, 'B': 1, 'F': 5},
    'D': {'B': 1, 'E': 1},
    'E': {'B': 4, 'D': 1, 'F': 1},
    'F': {'C': 5, 'E': 1, 'G': 1},
    'G': {'F': 1},
}
H = Graph.from_adjacency_list(AL, directed=False) # type: ignore

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
