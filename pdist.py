import random
from collections import defaultdict
import operator as op

def droplow(ndrop=1):
    def dropnlow(nums):
        return tuple(sorted(nums)[ndrop:])
    return dropnlow
def drophigh(nums, ndrop=1):
    def dropnhigh(nums):
        return tuple(sorted(nums)[:-ndrop])
    return dropnhigh
def adv(nums):
    return max(nums)
def disadv(nums):
    return min(nums)
def reliable(nums, av=6):
    return None
def sumt(nums):
    return (sum(nums),)

class PDist(object):
    def __init__(self, outcomes):
        """PDist is a probability distribution. Provide a list of outcomes (which will be assigned equal weights)
        or a dictionary of the form {outcome: weight, ...}

        Args:
            outcomes (list|dict): As above
        """
        if isinstance(outcomes, (list,range)):
            outcomes = {(outcome,): 1 for outcome in outcomes}
        self._outcomes = {outcome:weight for outcome, weight in outcomes.items()}
    
    @classmethod
    def dice_roll(cls, n):
        return cls(list(range(1,n+1)))

    @property
    def total_weight(self) -> float:
        """The total weight"""
        return sum([i for i in self._outcomes.values()])
    
    def normalise(self) -> None:
        """Rescale all the weights so that they sum to 1"""
        tot = self.total_weight
        self._outcomes = {outcome:weight/tot for outcome, weight in self._outcomes.items()}

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return PDist({outcome+other:weight for outcome, weight in self._outcomes.items()})
        return PDist.combine(self, other)

    def __repr__(self):
        t=self.total_weight
        return f"PDist({{{', '.join([f'{a}:{b/t*100:.2f}%' for a,b in self._outcomes.items()])}}})"

    @staticmethod
    def combine(a: "PDist", b: "PDist") -> "PDist":
        """Combine two individual PDists, with the outcome becoming the cartesian product of the two."""
        outc = []
        # outcomes = {}
        tota = a.total_weight
        totb = b.total_weight
        for out_a, w_a in a._outcomes.items():
            for out_b, w_b in b._outcomes.items():
                outc.append(((*out_a, *out_b), w_a*w_b/(tota*totb)))
        return PDist({a:w for a,w in outc})
        # for a, b, w in outc:
        #     if a+b in outcomes:
        #         outcomes[a+b] += w
        #     else:
        #         outcomes[a+b] = w
        # return PDist(outcomes)
    
    def map_outcomes(pdist: "PDist", f=sumt) -> "PDist":
        """Take a PDist and map the 'outcome' according to some function f. Some possibilities involve:
        pdist.adv - takes the max of the outcomes
        pdist.disadv - takes the min of the outcomes
        pdist.drophigh - drop the highest n outcomes
        pdist.droplow - drop the lowest n outcomes
        pdist.sum - combine all the outcomes to a single value

        Returns:
            PDist: as the input, but {outcome: weight,... } -> {f(outcome): weight}
        """
        output: dict = defaultdict(float)
        for outcome in pdist._outcomes:
            # fo = f(outcome)
            # if fo in output:
            output[f(outcome)] += pdist._outcomes[outcome]

        return PDist(output)

    def reduce_outcomes(self):
        self = self.map_outcomes()

    def select(self, n=1, rng: random.Random=None) -> list:
        if rng is None:
            rng = random.Random()
        return rng.choices([i for i in self._outcomes.keys()], [i for i in self._outcomes.values()], k=n)
    
    def min(self):
        return min(map(sum, self._outcomes))
    def max(self):
        return max(map(sum, self._outcomes))

    def average(self):
        totw = sum([w for w in self._outcomes.values()])
        return sum([sum(outc)*weight for outc, weight in self._outcomes.items()])/totw

    def plot(self):
        p = self.map_outcomes()
        from matplotlib import pyplot as plt
        plt.bar(list(map(sum, p._outcomes.keys())), p._outcomes.values())
        plt.show()

    def aplot(self, ysc=15, pad_h = 3):
        pad=pad_h
        def x_axis(n=6, pad=3):
            ticks = ('-'*(pad-1)+'+')*n
            labels = ''.join([f'{i+1:{pad}}' for i in range(n)])
            return ticks, labels
        dy = max(self._outcomes.values()) / ysc
        cy = 0
        cont=True
        rows = []
        while cont:
            cont=False
            cy += dy
            row = ['#' if i>=cy else ' ' for i in self._outcomes.values()]
            if "#" in row:
                cont = True
            rows.append(row)
        print('^')
        for row in reversed(rows):
            print('|' + (' '*(pad-1)) + ''.join([f'{dit:{pad}}' for dit in row]))
        axis, labels = x_axis(n=len(self._outcomes), pad=pad)
        print('+' + axis + '->')
        print(' ' + labels)

def crit(pdist: PDist, crit_on=20, max_on=None):
    roll_twice = (pdist+pdist).map_outcomes()
    # print('roll_twice: ', roll_twice)
    new_outcomes = list(range(pdist.min(), 2*pdist.max()+1))
    # print('all outcomes:', new_outcomes)
    new_pdist = PDist(new_outcomes)
    pdist.normalise()
    if max_on is None:
        max_chance = 0
        crit_chance = (21-crit_on)/20
        norm_chance = (crit_on-1)/20
    else:
        max_chance = (21-max_on)/20
        crit_chance = (max_on-crit_on)/20
        norm_chance = (crit_on-1)/20
    max_outcome = max(new_pdist._outcomes.keys())
    for outcome in new_pdist._outcomes.keys():
        n_prob = pdist._outcomes.get(outcome, 0)
        c_prob = roll_twice._outcomes.get(outcome, 0)
        if outcome == max_outcome:
            m_prob=1
        else:
            m_prob=0
        # print(outcome, ': ', n_prob, c_prob)
        new_pdist._outcomes[outcome] = norm_chance*n_prob + crit_chance * c_prob + max_chance * m_prob
    return new_pdist



# if __name__ == "__main__":
d4 = PDist.dice_roll(4) # ([1,2,3,4])
d6 = PDist.dice_roll(6) # ([1,2,3,4,5,6])
d8 = PDist.dice_roll(8) # ([1,2,3,4,5,6,7,8])
d10 = PDist.dice_roll(10) # ([1,2,3,4,5,6,7,8,9,10])
d12 = PDist.dice_roll(12) # ([1,2,3,4,5,6,7,8,9,10,11,12])

d4_2 = (d4+d4).map_outcomes()
d6_2 = (d6+d6).map_outcomes()
d8_2 = (d8+d8).map_outcomes()
d10_2 = (d10+d10).map_outcomes()
d12_2 = (d12+d12).map_outcomes()

d4_adv = (d4+d4).map_outcomes(adv)
d4_disadv = (d4+d4).map_outcomes(disadv)
d6_adv = (d6+d6).map_outcomes(adv)
d6_disadv = (d6+d6).map_outcomes(disadv)
d8_adv = (d8+d8).map_outcomes(adv)
d8_disadv = (d8+d8).map_outcomes(disadv)
d10_adv = (d10+d10).map_outcomes(adv)
d10_disadv = (d10+d10).map_outcomes(disadv)
d12_adv = (d12+d12).map_outcomes(adv)
d12_disadv = (d12+d12).map_outcomes(disadv)


    