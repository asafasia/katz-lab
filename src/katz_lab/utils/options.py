from dataclasses import dataclass


@dataclass
class Options:
    n_avg: int = 1000
    state_discrimination: bool = True
    plot: bool = True
    simulate: bool = False
