from typing import Tuple
from .optimizer import _params_t, Optimizer

class NRMSprop(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., irho: float=..., col: int=..., alpha: float=..., eps: float=..., weight_decay: float=..., momentum: float=...,  centered: bool=...) -> None: ...
