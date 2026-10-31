import numpy as np


class Particle:
    def __init__(self, x):
        self._x = np.array(x)
        self._v = np.zeros_like(x)
        self._x_best = np.array(x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_value):
        self._x = new_value

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_value):
        self._v = new_value

    @property
    def x_best(self):
        return self._x_best

    @x_best.setter
    def x_best(self, new_value):
        self._x_best = new_value
