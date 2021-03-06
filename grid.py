from __future__ import annotations
import math
from collections import defaultdict, namedtuple
from threading import current_thread
from heuristics import Heuristic
import uuid

from utils.priority_queue import PriorityQueue

Edge = namedtuple("Edge", ('weight'))

class NodeError(IndexError):
    pass

class NeighbourError(NodeError):
    pass

class NodeNotFoundError(NodeError):
    pass

class Node(object):
    """Node class

    Args:
        object (iden): Identifier for the node. Must be unique, must be hashable. Maybe use an integer coordinate? 
    """
    def __init__(self, iden):
        self.iden=iden
        self.neighbours = defaultdict(dict)
    def __lt__(self, other) -> bool:
        return self.iden < other.iden
    def __eq__(self, other) -> bool:
        """Two nodes are equal if their identifiers are equal. 

        Args:
            other (Node): The other node

        Returns:
            Boolean: True if the identifiers are equal
        """
        return self.iden == other.iden
    def __hash__(self) -> int:
        return hash(self.iden)
    def add_neighbour(self, neighbour: Node, weight_forwards: float=1, weight_backwards:float=None, strict=False):
        """Adds a neighbour `neighbour` to the node with a bi-directional edge
        Cases: 
         - no weights specified: Bi-directional edge with weight 1 both directions
         - weight_fowards=int, weight_backwards=None: 


        Args:
            neighbour (Node): The neighbour node
            weight_forwards (int, optional): Weight for the forwards direction. Defaults to 1.
            weight_backwards (int, optional): Weight for the backwards direction. If not specified, will use forwards direction. Defaults to None.
            strict (boolean, optional): If strict mode, IndexError raised if the neighbour is already a neighbour. Defaults to False.

        Raises:
            IndexError: Raised if an existing neighbour is added to the node (if strict=True)
        """
        if self.is_neighbour(neighbour):
            if strict:
                raise NeighbourError("Already a neighbour")
            else:
                return
        # self.neighbours[neighbour.iden] = dict()
        if weight_backwards is None:
            weight_backwards = weight_forwards
        self.neighbours[neighbour.iden]['weight'] = weight_forwards
        self.neighbours[neighbour.iden]['neighbour'] = neighbour
        neighbour.neighbours[self.iden]['weight'] = weight_backwards
        neighbour.neighbours[self.iden]['neighbour'] = self
        # self.neighbours.append((neighbour,weight))
        # neighbour.neighbours.append((self, weight))
    def set_weight(self, neighbour: Node, weight_forwards, set_reverse=True,weight_backwards=None):
        """Modify the weight of an edge.

        Args:
            neighbour (Node): The neighbouring node to modify
            weight_forwards (num): The forwards weight
            set_reverse (bool, optional): Set true in order to force the reverse weight to also be set. Defaults to True.
            weight_backwards (num, optional): The reverse weight. If not set, defaults to weight_forwards.

        Raises:
            NeighbourError: Error raised if the supplied neighbour is not already a neighbour.
        """
        if not self.is_neighbour(neighbour):
            raise NeighbourError(f"{self.iden} does not have {neighbour.iden} as a neighbour")
        if set_reverse:
            if not neighbour.is_neighbour(self):
                raise NeighbourError(f"{neighbour.iden} does not have {self.iden} as a neighbour (reverse)")
            if weight_backwards is None:
                weight_backwards = weight_forwards
        self.neighbours[neighbour.iden]['weight'] = weight_forwards
        if set_reverse:
            neighbour.neighbours[self.iden]['weight'] = weight_backwards
            # neighbour.neighbours[self.iden]['neighbour'] = self

    def add_neighbour_oneway(self, neighbour: Node, weight=1, strict=False):
        if self.is_neighbour(neighbour):
            if strict:
                raise NeighbourError("Already a neighbour")
            else:
                return
        self.neighbours[neighbour.iden]['weight'] = weight
        self.neighbours[neighbour.iden]['neighbour'] = neighbour
        
    def is_neighbour(self, neighbour: Node) -> bool:
        return neighbour.iden in self.neighbours

    def __repr__(self):
        return f"Node({self.iden})"

class NodeMap(object):
    """NodeMap at its most simplest is a container for all the nodes. It 
    allows Nodes to inherit coordinates and a heuristic function to facilitate 
    calculation of h-scores.
    NodeMap.nodes = {iden: {node:node, ... }, 
                     iden: {node:node, ... }, ... }
    NodeMap.heuristic = f(node, node) = num

    This is also where you'd put any dynamic node generation code.
    """
    def __init__(self):
        pass
    def get(self, iden):
        """Get a node by id. Must be implemented

        Args:
            iden (identifier): Any hashable identifier for a node

        Returns:
            node: node corresponding to iden if found, otherwise None
        """
        return None
    def heuristic(self, node_a_iden, node_b_iden):
        """Heuristic function, should take two node IDs and return the distance between them (or weight)

        Args:
            node_a_iden (NodeID): Node 1
            node_b_iden (NodeID): Node 2

        Returns:
            heuristic: The distance (or heuristic) between the input points.
        """
        return None
    

class SquareGrid(NodeMap):
    """A NodeMap made up of a square grid. Nodes are assigned iden of the tuple (x,y)

    Args:
        NodeMap ([type]): [description]
    """
    def __init__(self, width, height, walls=[], heuristic=Heuristic.grid_dist):
        self.height = height
        self.width = width
        self.walls = walls
        self.nodes = dict() #defaultdict(namedtuple('NodeMap', ('node', 'x','y')))
        self.coordinates = dict()
        # self.new_node = namedtuple('NodeDetails', ('node', 'x','y'))
        self._heuristic = heuristic
        self.generate_nodes()

    def heuristic(self, node_a, node_b):
        a = self.coordinates[node_a.iden]
        b = self.coordinates[node_b.iden]
        return self._heuristic(a,b)
    
    def get(self, iden):
        try:
            return self.nodes[iden] #.node
        except KeyError as e:
            raise NodeNotFoundError(f"There is no node with id {iden}")
        except TypeError as e:
            return None
    
    def get_coordinates(self, iden):
        return self.coordinates[iden]

    def get_or_create_node(self, x, y):
        iden = (x,y)
        try:
            n = self.get(iden)
        except NodeNotFoundError as e:
            n = Node(iden)
            self.add_node(n, x, y)
        return n
    def add_node(self, node, x, y):
        self.nodes[node.iden] = node #  self.new_node(node, x, y)
        self.coordinates[node.iden] = (x,y)

    def generate_nodes(self):
        for y in range(self.height):
            for x in range(self.width):
                if (x,y) in self.walls:
                    continue
                n = self.get_or_create_node(x,y)
                # self.calculate_neighbours(n)
                for (nx, ny), weight in self.all_dirs(x,y):
                    if not (0<=nx<self.width and 0<=ny<self.height):
                        continue
                    if (nx,ny) in self.walls:
                        continue
                    neighbour = self.get_or_create_node(nx, ny)
                    n.add_neighbour(neighbour,weight)
    
    # def calculate_neighbours(self, node):
    #     pass

    def all_dirs(self, x,y):
        sq2 = math.sqrt(2)
        yield from [((x+0,y+1),1), ((x+1,y+1),sq2),
                    ((x+1,y+0),1), ((x+1,y-1),sq2),
                    ((x+0,y-1),1), ((x-1,y-1),sq2),
                    ((x-1,y+0),1), ((x-1,y+1),sq2)]
    def ascii_print(self, path=None):
        print("___" * self.width)
        for y in range(self.height):
            for x in range(self.width):
                r = ' . '
                try: 
                    node = self.get((x,y))
                    if node in path:
                        r = ' @ '
                except:
                    r = '###'
                # if node.iden in self.walls:
                #     r = '###'
                print(r, end="")
            print()

class PathScores(object):
    def __init__(self):
        pass
    def g(self, g):
        self.g = g
    def h(self, h):
        self.h = h
    @property
    def f(self):
        return self.g+self.h

class A_Star(object):
    def __init__(self, node_map, start, finish):
        self.node_map = node_map
        self.open = PriorityQueue()
        self.open.put(start, 0)
        self.came_from = {start.iden:None}
        self.cost_so_far = {start.iden:0}
        self.finish = finish
        self.start = start

    def run(self):
        while not self.open.is_empty():
            active_node = self.open.get()
            if active_node == self.finish:
                break
            for neighbour_iden, attr in active_node.neighbours.items():
                # print(attr)
                neighbour = attr['neighbour']
                new_cost = self.cost_so_far[active_node.iden] + attr['weight']
                if neighbour.iden not in self.cost_so_far or new_cost < self.cost_so_far[neighbour.iden]:
                    self.cost_so_far[neighbour.iden] = new_cost
                    priority = new_cost + self.node_map.heuristic(neighbour, self.finish)
                    self.open.put(neighbour, priority)
                    self.came_from[neighbour.iden] = active_node
    def reconstruct(self):
        current = self.finish
        path = []
        while current != self.start:
            path.append(current)
            current = self.came_from[current.iden]
        path.append(self.start)
        path.reverse()
        return path



walls = [(1,1),(1,2),(1,3),(2,1),(3,1)]

walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
g = SquareGrid(10,10, walls=walls)#, heuristic=Heuristic.l1_norm)
start = g.get((1,9))
ne = g.get((2,9))
start.set_weight(ne, 2)

p = A_Star(g, g.get((1,9)), g.get((8,3)))
p.run()
g.ascii_print(path=p.reconstruct())
print(f"Cost: {p.cost_so_far[p.finish.iden]}")
# print(p.reconstruct())
