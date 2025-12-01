from dataclasses import dataclass


@dataclass
class Option:
    n_avg: int = 1000
    state_discrimination: bool = True
