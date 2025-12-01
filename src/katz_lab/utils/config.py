from pathlib import Path
import json

ROOT = Path.home() / "Janis Lab Code repos" / "Guy" / "janis-lab-opx+"

EXPERIMENT_UTILS = ROOT / "experiment_utils"
ARGS_PATH = EXPERIMENT_UTILS
OPTIMAL_WEIGHTS_PATH = EXPERIMENT_UTILS / "optimal_weights.npz"

sa_address = "TCPIP0::192.168.43.100::inst0::INSTR"
qm_host = "192.168.43.253"
qm_port = 9510


with open(ARGS_PATH, "r") as file:
    args = json.load(file)
