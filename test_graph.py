import unittest
from graph import *

class TestGraph(unittest.TestCase):
    def test_node(self):
        a = Node('a')
        b = Node('b')
        a.join(b, 1)
        self.assertEqual(a.get_weight(b), 1, "to see if node weights are added correctly.")
        self.assertIsNone(b.get_weight(a))

        b.join(a, 2)
        self.assertEqual(b.get_weight(a), 2)
        
        self.assertEqual(len(a),1)
        

    def test_graph(self):
        g = Graph(weighted=True, directed=True) ; a = g.add_node('a') ; b = g.add_node('b')
        
        g.add_edge('a','b',1,2) # Adds an edge from a->b with weight 1 AND b->a with weight 2

        self.assertTrue('a' in g, f"Error, 'a' not in graph (search by id)")
        self.assertTrue('b' in g, f"Error, 'b' not in graph (search by id)")

        self.assertTrue(a in g, f"Error, {a} not in graph (search by node)")
        self.assertTrue(b in g, f"Error, {b} not in graph (search by node)")

        # self.assertTrue(a[b]==1)
        # self.assertTrue(b[a]==2)

        h = Graph(weighted=False, directed=True) ; h.add_node('a') ; h.add_node('b')
        h.add_edge('a','b') # No weight needed, adds an unweighted edge a->b
        h.add_edge('b','a') # Again, no weight needed, adds an unweighted edge b->a

        i = Graph(weighted=True, directed=False) ; i.add_node('a') ; i.add_node('b')
        i.add_edge('a','b',3) # Adds an edge from a->b with weight 3 and from b->a with weight 3

        j = Graph(weighted=False, directed=False) ; j.add_node('a') ; j.add_node('b')
        j.add_edge('a','b') # Adds an unweighted edge a->b AND an unweight edge from b->a

    # def test_dates_between(self):
    #     rng = Random()
    #     start = '1970-01-01'
    #     end = '1970-01-31'
    #     n=1000
    #     d = util.dates_between(n,start,end,rng)
    #     c = Counter(d)
    #     d1 = dt.datetime.strptime(start,"%Y-%m-%d").date()
    #     d2 = dt.datetime.strptime(end,"%Y-%m-%d").date()
    #     # print(d1,d2,d1 in c, d2 in c)
    #     # print(c)
    #     self.assertTrue(d1 in c)
    #     self.assertTrue(d2 in c)
    # def test_ints(self):
    #     rng=Random()
    #     start=0
    #     end=100
    #     n=1000
    #     d = util.ints_between(n,start,end,rng)
    #     c = Counter(d)
    #     self.assertTrue(start in c)
    #     self.assertTrue(end in c)
    