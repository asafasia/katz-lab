# config_loader.py
import yaml
import socket
from pathlib import Path

def load_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def load_config(config_dir="config"):
    config_dir = Path(config_dir)

    # base config
    qpu_cfg = load_yaml(config_dir / "qpu.yaml")

    # per-qubit configs
    qubits = {}
    for qfile in (config_dir / "qubits").glob("*.yaml"):
        qdata = load_yaml(qfile)
        qubits[qdata["name"]] = qdata
    qpu_cfg["qubits"] = qubits

    # machine-local config
    hostname = socket.gethostname()
    local_cfg = load_yaml(config_dir / f"local_{hostname}.yaml")

    # merge dictionaries
    full = {**qpu_cfg, **local_cfg}

    return full