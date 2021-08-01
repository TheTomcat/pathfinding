# import pandas as pd
# import numpy as np
# based on https://towardsdatascience.com/generating-fake-data-with-pandas-very-quickly-b99467d4c618

from itertools import cycle
from random import Random
import datetime as dt
import adj

### Constants

default_categories = {
    'animals': adj.animals, #['cow', 'rabbit', 'duck', 'shrimp', 'pig', 'goat', 'crab', 'deer', 'bee', 'sheep', 'fish', 'turkey', 'dove', 'chicken', 'horse'],
    'names'  : ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth', 'Ahmed', 'Barbara', 'Richard', 'Susan', 'Salomon', 'Juan Luis'],
    'cities' : ['Stockholm', 'Denver', 'Moscow', 'Marseille', 'Palermo', 'Tokyo', 'Lisbon', 'Oslo', 'Nairobi', 'Río de Janeiro', 'Berlin', 'Bogotá', 'Manila', 'Madrid', 'Milwaukee'],
    'colors' : adj.colours #['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'pink', 'silver', 'gold', 'beige', 'brown', 'grey', 'black', 'white']
}

def make_username(rng: Random, dig=4) -> str:
    adjectives = adj.adjectives
    nouns = adj.nouns
    return rng.choice(adjectives).capitalize() + rng.choice(nouns).capitalize() + ''.join([str(rng.randint(0,9)) for d in range(dig)])

default_ranges = {
    'i': (0, 10),
    'f': (0, 100),
    'c': ('names',5),
    'd': ('1970-01-01','2021-12-31'),
    't': ('00:00:00', '23:59:59'),
    'b': (0.5, 0.5),
    'm': (make_username,{'dig':4})
}

suffix = {"c" : "cat", "i" : "int", "f" : "float", "d" : "date", "t":"time", "b":"bool", 'm':'misc'}#, "p":"point"}

### Generator functions
def dates_between(n: int, start: str, end: str, rng: Random, tformat: str="%Y-%m-%d") -> list:
    """Given a start date, end date and a number in (0,1) return n date objects in the range.

    Args:
        n (int): The number of date objects to return
        start (str): The start date
        end (str): The end date
        rng (Random): A random.Random instance
        format (str, optional): The date string format. Defaults to "%Y%m%d".

    Returns:
        datetime.date: The date between start and end
    """
    d1 = dt.datetime.strptime(start, tformat).date()
    d2 = dt.datetime.strptime(end, tformat).date()
    # i = (d2-d1).days * rng.uniform(0,1)
    return [d1 + dt.timedelta(days=rng.randint(0,(d2-d1).days)) for i in range(n)]

def ints_between(n: int, start: int, end: int, rng: Random) -> list:
    """Given a start and end, generate n integers in the range [start, end] inclusive.

    Args:
        n (int): The number of integers to return
        start (int): The start number
        end (int): The end number
        rng (Random): A random.Random instance

    Returns:
        list: A list of random integers uniformly distributed in the range [start, end]
    """
    return [rng.randint(start, end) for i in range(n)]

def floats_between(n: int, start: float, end: float, rng: Random) -> list:
    """Given a start and end, generate n numbers in the range [start, end].

    Args:
        n (int): The number of integers to return
        start (float): The start number
        end (float): The end number
        rng (Random): A random.Random instance

    Returns:
        list: A list of random floats uniformly distributed in the range [start, end]
    """
    return [rng.uniform(start, end) for i in range(n)]

def bools_distr(n: int, true_weight: float, false_weight: float, rng: Random) -> list:
    """Given a true_weight and false_weight, generate n boolean values according to that distribution

    Args:
        n (int): The number of values to return
        true_weight (float): The proportion of true values
        false_weight (false): The proportion of false values
        rng (Random): A random.Random instance

    Returns:
        list: A list of boolean values generated in the ration true_weight:false_weight
    """
    prob_true = true_weight/(true_weight+false_weight) 
    return [rng.uniform(0,1) < prob_true for i in range(n)]

def categorical(n: int, category_data: list, rng: Random) -> list:
    return rng.choices(category_data, k=n)

def generate_fake_data(n: int, columns: str, column_names=None, ranges_override=None, seed:int=None):
    """Generates filler data.
    i: Integer.     range: (min,max) defaults to (0,10) (inclusive)
    f: Float        range: (min,max) default to (0,100)
    c: Category     range: (str_first_column, cycle_length) defaults to ('name', 15) 
                        str_first_column should be one of ['animals','names','cities','colours'] and 
                        will mean that the first category item will have that category. The generator will
                        select from a total of cycle_length items as potential outputs. 

                        If provided as a range_override dict, instead you have the option of giving a list
                        of items to make up a category 
    t: Time         Time
    d: Date         Default range (1970-1-1 to 2021-12-31)
    b: Boolean      range: (true_probability, false_probability) defaults to (0.5,0.5)
    m: Misc         range: (func(rng: random.Random, *args) -> obj, args). Defaults to random username

    column_names should be a list of strings representing the headings of each column. If it is supplied
                it should be the same length as the number of columns. 

    ranges_override can either be a list or a dict object. 
        If it is a list, it must be the same length as the number of columns. 
        Each element should be a 2-tuple corresponding to the appropriate column definition as listed above.

        If it is a dict, the keys should be of the values 'ifctdb' and the values should be 2-tuples for 
        each column. 

    Args:
        n (int): The number of rows to generate
        columns (str): a string of [ifctdbl] representing the types of data
        column_names (list[str,...], optional): The column headings. If supplied, must be of length n.
        ranges_override ([list | dict], optional): [description]. Defaults to None.
        seed (int, optional): See supplied to the random number generator.
    """
    rng = Random(seed)
    
    first_c = default_ranges["c"][0] # What is the first category 
    categories_cycler = cycle([first_c] + [c for c in default_categories.keys() if c != first_c])

    # Handle any column names. 
    if isinstance(column_names,list): # If the user has supplied, they should have given the right amount
        assert len(column_names) == len(columns), f"The fake DataFrame should have {len(columns)} columns but col_names is a list with {len(column_names)} elements"
    elif column_names is None: # If the user has not supplied, make them up
        column_names = [f"col_{str(i)}_{suffix.get(col)}" for i, col in enumerate(columns)]

    # Handle the ranges_overrides
    if isinstance(ranges_override,list): # If the user has supplied, is it a list? It should be the same size as columns
        assert len(ranges_override) == len(columns), f"The fake DataFrame should have {len(columns)} columns but intervals is a list with {len(ranges_override)} elements"
        ranges_list = ranges_override # The ranges are provided directly by the user
    else:
        if isinstance(ranges_override,dict): # If it's a dict
            assert len(set(ranges_override.keys()) - set(default_ranges.keys())) == 0, f"The intervals parameter has invalid keys"
            updated_ranges = {key: val for key, val in default_ranges.items()}.update(ranges_override)
            # The only valid keys are 'ifdtbc'
        else:
            updated_ranges = {key: val for key, val in default_ranges.items()}
        updated_ranges['c'] = (categories_cycler, updated_ranges['c'][1])
        ranges_list = [updated_ranges[col] for col in columns] # The default ranges have been updated by the user
        
    df = {}
    for column_type, column_name, interval in zip(columns, column_names, ranges_list):
        if column_type == 'i': # int
            start, end = interval
            df[column_name] = ints_between(n, start, end, rng) 
            # [rng.randint(start, end) for i in range(n)]
        elif column_type == 'f': # float
            start, end = interval
            df[column_name] = floats_between(n, start, end, rng) 
            # [rng.uniform(start, end) for i in range(n)]
        elif column_type == 'd': # date
            start, end = interval
            df[column_name] = dates_between(n, start, end, rng)
        elif column_type == 't': # time
            start, end = interval
        elif column_type == 'b': # boolean
            t_w, f_w = interval
            df[column_name] = bools_distr(n, t_w, f_w, rng)
        elif column_type == 'm': # misc
            f, kwargs = interval
            df[column_name] = [f(rng, **kwargs) for i in range(n)]
        elif column_type == 'c': # category
            if isinstance(interval, list):
                # This is actually a list of items in a category, so we'll just choose from the user-supplied list
                category_data = interval
            else:
                category_name, pop_size = interval
                
                # The only other option is a user-supplied 2-tuple. (name, pop_size)
                if isinstance(category_name, cycle):
                    category_name = next(category_name)
                assert category_name in default_categories, f"There are no samples for the category '{category_name}'. Available categories: {default_categories.keys()}"
                if pop_size == -1:
                    pop_size = len(default_categories[category_name])
                try:
                    category_data = rng.sample(default_categories[category_name], k=pop_size)
                except:
                    category_data = default_categories[category_name]
            df[column_name] = categorical(n, category_data, rng)
    return df

p = generate_fake_data(1000, 'cicfdbm', 
    column_names=[
        "Name",
        "Age",
        "Pet",
        "Height",
        "Birthday",
        "Has Sibling",
        "Username"
    ],
    ranges_override=[
        ('names',-1),
        (0,100),
        ('animals',15),
        (150,210),
        ("1970-01-01","2021-12-31"),
        (0.5,0.5),
        (make_username,{'dig':4})
    ],
seed=1)