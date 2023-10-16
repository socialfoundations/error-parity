import operator
from functools import reduce


def join_dictionaries(*dicts) -> dict:
    return reduce(operator.or_, dicts)
