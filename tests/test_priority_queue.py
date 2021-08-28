import unittest
from utils import priority_queue

class QueueTest(unittest.TestCase):
    def test_put(self):
        q = priority_queue.Queue()
        q.put(1)
        q.put(2)
        q.putleft(3)
        self.assertEqual(q.get(), 3) # putleft
        self.assertEqual(q.get(), 1) # FIFO list
        self.assertEqual(q.get(), 2)
        self.assertTrue(q.is_empty()) # And now empty
        self.assertRaises(IndexError, q.get)

class PriorityQueueTest(unittest.TestCase):
    def test_prorityq_putpop(self):
        q = priority_queue.PriorityQueue()
        q.put('a',0)
        q.put('b',1)
        q.put('c',10)
        q.put('d',3)
        self.assertEqual(q.get(), 'a') # putleft
        self.assertEqual(q.get(), 'b') # FIFO list
        self.assertEqual(q.get(), 'd')
        self.assertEqual(q.get(), 'c')
        self.assertTrue(q.is_empty()) # And now empty
        self.assertRaises(IndexError, q.get)
