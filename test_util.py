import unittest
import util
from collections import Counter
from random import Random
import datetime as dt

class TestUtil(unittest.TestCase):
    def test_dates_between(self):
        rng = Random()
        start = '1970-01-01'
        end = '1970-01-31'
        n=1000
        d = util.dates_between(n,start,end,rng)
        c = Counter(d)
        d1 = dt.datetime.strptime(start,"%Y-%m-%d")
        d2 = dt.datetime.strptime(end,"%Y-%m-%d")
        self.assertTrue(d1 in c)
        self.assertTrue(d2 in c)
        