import math
from collections import defaultdict, namedtuple

Edge = namedtuple("Edge", ('weight'))

class Node(object):
    """Node class

    Args:
        object (iden): Identifier for the node. Must be unique, must be hashable. Maybe use an integer coordinate? 
    """
    def __init__(self, iden):
        self.iden=iden
        self.neighbours = defaultdict(dict)
    def __eq__(self, other):
        """Two nodes are equal if their identifiers are equal. 

        Args:
            other (Node): The other node

        Returns:
            Boolean: True if the identifiers are equal
        """
        return self.iden == other.iden
    def __hash__(self):
        return hash(self.iden)
    def add_neighbour(self, neighbour, weight_forwards=1, weight_backwards=None, strict=False):
        """Adds a neighbour `neighbour` to the node with a bi-directional edge
        Cases: 
         - no weights specified: Bi-directional edge with weight 1 both directions
         - weight_fowards=int, weight_backwards=None: 


        Args:
            neighbour ([type]): The neighbour node
            weight_forwards (int, optional): Weight for the forwards direction. Defaults to 1.
            weight_backwards (int, optional): Weight for the backwards direction. If not specified, will use forwards direction. Defaults to None.
            strict (boolean, optional): If strict mode, IndexError raised if the neighbour is already a neighbour. Defaults to False.

        Raises:
            IndexError: Raised if an existing neighbour is added to the node (if strict=True)
        """
        if self.is_neighbour(neighbour):
            if strict:
                raise IndexError("Already a neighbour")
            else:
                return
        if weight_backwards is None:
            weight_backwards = weight_forwards
        self.neighbours[neighbour]['weight'] = weight_forwards
        neighbour.neighbours[self]['weight'] = weight_backwards
        # self.neighbours.append((neighbour,weight))
        # neighbour.neighbours.append((self, weight))
    def add_neighbour_oneway(self, neighbour, weight=1):
        if self.is_neighbour(neighbour):
            return
        self.neighbours[neighbour]['weight'] = weight
    def is_neighbour(self, neighbour):
        return neighbour in self.neighbours
    def __repr__(self):
        return f"Node({self.iden})"

class NodeMap(object):
    def __init__(self):
        pass
    def calculate_neighbours(self, node):
        return None
    

class SquareGrid(NodeMap):
    def __init__(self, width, height, walls=[]):
        self.height = height
        self.width = width
        self.walls = walls
        self.nodes = []
        self.generate_nodes()
    
    def get_node_with_id(self, iden):
        try:    
            return next(filter(lambda x: x.iden==iden, self.nodes))
        except StopIteration:
            return None

    def get_or_create_node_with_id(self, iden):
        n = self.get_node_with_id(iden)
        if n is None:
            n = Node(iden)
            self.nodes.append(n)
        return n

    def generate_nodes(self):
        for y in range(self.height):
            for x in range(self.width):
                n = self.get_or_create_node_with_id((x,y))
                # self.calculate_neighbours(n)
                for (nx, ny), weight in self.all_dirs(x,y):
                    if not (0<=nx<self.width and 0<=ny<self.height):
                        continue
                    if (nx,ny) in self.walls:
                        continue
                    neighbour = self.get_or_create_node_with_id((nx, ny))
                    n.add_neighbour(neighbour,weight)
    
    def calculate_neighbours(self, node):
        x,y = node.iden
        for (nx, ny), weight in self.all_dirs(x,y):
            if not (0<=nx<self.width and 0<=ny<self.height):
                continue
            if (nx,ny) in self.walls:
                continue
            neighbour = self.get_or_create_node_with_id((nx, ny))
            node.add_neighbour(neighbour,weight)

    def all_dirs(self, x,y):
        sq2 = math.sqrt(2)
        yield from [((x+0,y+1),1), ((x+1,y+1),sq2),
                    ((x+1,y+0),1), ((x+1,y-1),sq2),
                    ((x+0,y-1),1), ((x-1,y-1),sq2),
                    ((x-1,y+0),1), ((x-1,y+1),sq2)]

walls = [(1,1),(1,2),(1,3),(2,1),(3,1)]

g = SquareGrid(5,5, walls=walls)

class Heuristic(object):
    @staticmethod 
    def l2_norm(a, b):
        x1,y1 = a.iden
        x2,y2 = b.iden
        return math.sqrt((x2-x1)**2+(y2-y1)**2)
    @staticmethod
    def l1_norm(a,b):
        x1,y1 = a.iden
        x2,y2 = b.iden
        return abs(x1-x2)+abs(y1-y2)
    @staticmethod
    def grid_dist(a,b):
        x1,y1 = a.iden
        x2,y2 = b.iden
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        low = min(dx,dy)
        dd = abs(dy-dx)
        return low*math.sqrt(2) + dd

def f(d):
    d['f'] = d['g'] + d['h']
    return d

class A_Star(object):
    def __init__(self, NodeMap, start, finish, heuristic=Heuristic.euclidean):
        self.open = []
        self.closed = []
        self.heuristic = heuristic
        self.open[start] = f({'g':0, 'h':self.heuristic(start, finish)})
        self.finish = finish
    # def run(self):
    #     loop
    #         current = min(self.open) -> pop
    #         closed.append(current)
    #         if current == self.finish:
    #             return
    #         for neighbour in current.neighbours:
    #             if neighbour in closed:
    #                 continue
    #             if neighbour not in open or path(current, neighbour) < neighbour.f:
    #                 neighbour.f
    #                 neighbour.parent=current
    #                 if neighbour not in open:
    #                     open.append(neighbour)


print(Heuristic.grid_dist())