from typing import Tuple, Iterable, Optional


class Algorithm:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)

    def run(self, **kwargs) -> Tuple[Tuple[Iterable, Iterable], Optional[Tuple]]:
        pass
