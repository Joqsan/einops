__author__ = 'Joqsan Azocar'

from typing import Any
import tinygrad

from . import RearrangeMixin, ReduceMixin


class Rearrange(RearrangeMixin):
    def __call__(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin):
    def __call__(self, input):
        return self._apply_recipe(input)