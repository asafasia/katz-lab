from dataclasses import dataclass, field


@dataclass
class Options:
    n_avg: int = 100
    state_discrimination: bool = True
    plot: bool = True
    simulate: bool = False
    save: bool = True
    dc_set_voltage: bool = False
    states_to_measure: list = field(default_factory=lambda: ["gef"])
    update_args: bool = False
    active_reset: bool = False
    active_reset_n: int = 2
    pulse_type: str = "gaussian"
    readout_method: str = "simple"
    
