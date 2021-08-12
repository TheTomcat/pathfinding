def is_documented_by(f):
  def wrapper(target):
    target.__doc__ = f.__doc__
    return target
  return wrapper