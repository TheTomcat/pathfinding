import unittest
import factorial

class TestFactor(unittest.TestCase):
    def test_iterable(self):
        iterate = factorial.PossibleFactors(1000)
        exp_output = [2] + [2*i+1 for i in range(1,16)]
        for i,j in zip(iterate, exp_output):
            self.assertEqual(i,j)
    
    def test_factorial(self):
        f = factorial.Factor(16)
        self.assertEqual(f._factors, {2:4})
        f = factorial.Factor(2**6*3**4*7)
        self.assertEqual(f._factors, {2:6,3:4,7:1})

    def test_factorial_mul(self):
        for i in range(2,200):
            for j in range(2,200):
                f = factorial.Factor(i)
                g = factorial.Factor(j)
                h = f*g
                self.assertEqual(h.eval(), i*j, f"Failed on {i*j}")
    
    def test_factorial_div(self):
        for i in range(2,200):
            for j in range(2,200):
                f = factorial.Factor(i)
                g = factorial.Factor(j)
                h = f/g
                self.assertEqual(round(h.eval(),8), round(i/j,8), f"Failed on {i}/{j}={h._factors}")

    def test_factorial_eval(self):
        for i in range(2,5000):
            self.assertEqual(i, factorial.Factor(i).eval(), f"Failed on {i}")