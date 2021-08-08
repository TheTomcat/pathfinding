import random
from collections import defaultdict

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
    
    @property
    def total_weight(self) -> float:
        """The total weight"""
        return sum([i for i in self._outcomes.values()])
    
    def normalise(self) -> None:
        """Rescale all the weights so that they sum to 1"""
        tot = self.total_weight
        self._outcomes = {outcome:weight/tot for outcome, weight in self._outcomes}

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
    
    def map_outcomes(pdist: "PDist", f=sum) -> "PDist":
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

    def select(self, n=1, rng: random.Random=None) -> list:
        if rng is None:
            rng = random.Random()
        return rng.choices([i for i in self._outcomes.keys()], [i for i in self._outcomes.values()], k=n)

def droplow(nums, ndrop=1):
    return sum(sorted(nums)[ndrop:])
def drophigh(nums, ndrop=1):
    return sum(sorted(nums)[:-ndrop])
def adv(nums):
    return max(nums)
def disadv(nums):
    return min(nums)

if __name__ == "__main__":
    d6 = PDist([1,2,3,4,5,6])
    d6_2 = d6+d6
    adv_d6 = d6_2.map_outcomes(adv)
    disadv_d6 = d6_2.map_outcomes(disadv)

    skills = (d6+d6+d6+d6).map_outcomes(droplow)