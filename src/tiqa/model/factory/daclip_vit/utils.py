import collections.abc
from itertools import repeat


def _ntuple(n):  # type: ignore
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
