from pathlib import Path
import katz_lab

import json

# ROOT = Path.home()

KATZ_LAB_PATH = Path(katz_lab.__file__).resolve().parent

ARGS_PATH = KATZ_LAB_PATH / "data" / "params" / "params.json"

OPTIMAL_WEIGHTS_PATH = KATZ_LAB_PATH / "data" / "kernel_traces" / "optimal_weights.npz"

qubit = "q10"


sa_address = "TCPIP0::192.168.43.100::inst0::INSTR"
qm_host = "192.168.43.253"
qm_port = 9510


