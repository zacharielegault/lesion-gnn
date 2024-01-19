from typing import Generic, TypeVar

T = TypeVar("T")


class Placeholder(Generic[T]):
    def __init__(self):
        self._value = None

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("Placeholder value not set")
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = value
