import enum


class SplitTypes(enum.Enum):
    NORMAL = 1
    BALANCED = 2


class Split:
    def __init__(self, type: SplitTypes, value: list = None):
        self._type = type
        self._value = value

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        return self._value
