__author__ = 'Joqsan Azocar'

from typing import Any
import tinygrad

from . import RearrangeMixin, ReduceMixin


class Rearrange(RearrangeMixin):
    def forward(self, input):
        return self._apply_recipe(input)

    def __call__(self, input):
        return self.forward(input)


class Reduce(ReduceMixin):
    def forward(self, input):
        return self._apply_recipe(input)

    def __call__(self, input):
        return self.forward(input)