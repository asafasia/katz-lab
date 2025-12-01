from pathlib import Path
import json

ROOT = Path.home()

KATZ_LAB_PATH = ROOT / "Developer" / "katz-lab" / "src" / "katz_lab"

ARGS_PATH = KATZ_LAB_PATH / "data" / "params" / "args_ariel.json"

OPTIMAL_WEIGHTS_PATH = KATZ_LAB_PATH / "data" / "kernel_traces" / "optimal_weights.npz"

qubit = "qubit10"


sa_address = "TCPIP0::192.168.43.100::inst0::INSTR"
qm_host = "192.168.43.253"
qm_port = 9510


with open(ARGS_PATH, "r") as file:
    args = json.load(file)


qubit_args = args[qubit]["qubit"]

if __name__ == "__main__":
    from pprint import pprint

    pprint(args)
