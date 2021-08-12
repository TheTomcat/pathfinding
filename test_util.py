import unittest
import dummy_data
from collections import Counter
from random import Random
import datetime as dt

class TestUtil(unittest.TestCase):
    def test_dates_between(self):
        rng = Random()
        start = '1970-01-01'
        end = '1970-01-31'
        n=1000
        d = dummy_data.dates_between(n,start,end,rng)
        c = Counter(d)
        d1 = dt.datetime.strptime(start,"%Y-%m-%d").date()
        d2 = dt.datetime.strptime(end,"%Y-%m-%d").date()
        # print(d1,d2,d1 in c, d2 in c)
        # print(c)
        self.assertTrue(d1 in c)
        self.assertTrue(d2 in c)
    def test_ints(self):
        rng=Random()
        start=0
        end=100
        n=1000
        d = dummy_data.ints_between(n,start,end,rng)
        c = Counter(d)
        self.assertTrue(start in c)
        self.assertTrue(end in c)
    