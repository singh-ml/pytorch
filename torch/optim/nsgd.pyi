from .optimizer import _params_t, Optimizer

class NSGD(Optimizer):
    def __init__(self, params: _params_t, lr: float, irho: float, col: int, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=...) -> None: ...
