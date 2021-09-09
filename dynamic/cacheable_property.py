import abc
from functools import wraps
import logging
import time

class CacheableMixin(abc.ABC):
    def mark_as_stale(obj, args=None):
        """Marks the refreshable properties contained within the class as stale. 

        Args:
            args (List[str], optional): An optional list of properties. If none are provided, mark them all as stale.
        """
        if args is None:
            lst = obj._refreshable_properties
        else:
            lst=args
        for arg in lst:
            setattr(obj, f"_{arg}_is_fresh", False)

class cacheable_property:
    """Adds the @cacheable_property decorator.

    This allows a class method to be cached and optionally recalculated as needed, when it
    becomes stale.

    Usage:
    class TestClass(object):
        @cacheable_property
        def an_expensive_function(self):
            time.sleep(4)
            return 2

    p = TestClass()
    p.an_expensive_function()
    ...
    >>> (delay) 4
    p.an_expensive_function()
    >>> 4

    Note that the refreshable_properties can force a recalculation as follows:
    p.an_expensive_function(force_update=True)

    The mixin RefreshableMixin adds mark_as_stale, which allows for properties to be marked
    as stale (and hence recalculated when next required).

    mark_as_stale takes an optional argument of a list of strings representing the 
    properties to be marked as stale. If this is not provided, all the refreshable properties
    will be marked as stale.
    """
    def __init__(self, target):
        self.fn = target

    def __set_name__(self, owner, name):
        logging.info(f"decorating {self.fn} and using {owner}, {name}")
        
        _refreshable_properties = getattr(owner, '_refreshable_properties', list())
        _refreshable_properties.append(name)
        setattr(owner, '_refreshable_properties', _refreshable_properties)

        property_name = name
        property_value = f"_{property_name}_val"
        fresh_prop = f'_{property_name}_is_fresh'
        recalculate_prop = f'_{property_name}_recalculate'
        
        def refresh(obj, *args, **kwargs):
            
            print(self.fn.__doc__)
            is_fresh = getattr(obj, fresh_prop)
            logging.info(f" -> fresh? {is_fresh}")
            if is_fresh and not kwargs.get('force_update', False):
                logging.info("so return val")
                return getattr(obj, property_value)
            logging.info("so return calculated val")
            return getattr(obj, recalculate_prop)()
        refresh.__doc__ = self.fn.__doc__
        def recalculate(obj):
            logging.info(f"Recalculating {name}")
            val = self.fn(obj)
            setattr(obj, fresh_prop, True)
            setattr(obj, property_value, val)
            return val

        setattr(owner, fresh_prop, False)
        setattr(owner, property_value, None)
        setattr(owner, recalculate_prop, recalculate)
        setattr(owner, property_name, refresh)

    def __call__(self, force_update=False):
        pass # TO avoid typechecking errors

        # setattr(owner, name, self.fn)
            # https://docs.python.org/3/reference/datamodel.html#object.__set_name__
            # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class


if __name__ == "__main__":
    class A:
        @cacheable_property
        def hello(self):
            time.sleep(2)
            return 4
        
        @cacheable_property
        def hey(self):
            time.sleep(2)
            return 2
    p=A()
    q=A()
    print(p.hey())
    print(p.hey())
    print(p.hey(force_update=True))
