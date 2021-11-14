from typing import Tuple
from .optimizer import _params_t, Optimizer

class N_Adam(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., irho: float=..., col: int=..., betas: Tuple[float, float]=..., eps: float=..., weight_decay: float=..., amsgrad: bool = ...) -> None: ...
