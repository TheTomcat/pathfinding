from itertools import product
from collections import defaultdict
from typing import List

n = 6
d = 6

rolls = list(product(range(1,d+1), repeat=n))
rerolls: List[tuple] = []

def recompute_rolls(rolls):
    rerolls = []
    for roll in rolls:
        reroll = []
        reroll.append(roll[0])
        for next_roll in roll[1:]:
            if next_roll in reroll:
                while next_roll in reroll and next_roll <= 6:
                    next_roll += 1
            reroll.append((next_roll-1) % d + 1)
        rerolls.append(reroll)
    return rerolls

def calculate_avg_time_till_first_roll(rerolls):
    times = defaultdict(list)
    for roll in rerolls:
        for position, value in enumerate(roll):
            times[value].append(position)
    return times
