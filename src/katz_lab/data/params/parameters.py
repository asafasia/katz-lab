import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# 1. Define leaf nodes (Gates, pulses)
class GateParams(BaseModel):
    length: int
    amplitude: float
    # Optional fields default to None if missing
    sigma: Optional[float] = None
    drag_coef: Optional[float] = None
    
class SquareGate(BaseModel):
    length: int
    length_ef: int
    length_gf: int
    amplitude_180: float
    amplitude_90: float
    amplitude_180_ef: float
    amplitude_180_gf: float

class IQPair(BaseModel):
    I: float
    Q: float

# 2. Define the Qubit structure
class QubitParams(BaseModel):
    IQ_input: IQPair
    IQ_bias: IQPair
    qubit_correction_matrix: List[float]
    qubit_LO: float
    qubit_ge_freq: float
    qubit_ef_freq: float
    qubit_anharmonicity: float
    
    # Nested objects
    reset_gate: GateParams
    square_gate: SquareGate
    gaussian_gate: GateParams
    cos_gate: GateParams
    drag_gate: GateParams
    saturation_pulse: GateParams
    
    T1: float
    T2: float
    thermalization_time: int
    
    # Example of validation: Ensure frequencies are positive
    @field_validator('qubit_ge_freq', 'qubit_ef_freq', 'qubit_LO')
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError('Frequency must be positive')
        return v

# 3. Define the Resonator structure
class ResonatorParams(BaseModel):
    IQ_input: IQPair
    IQ_bias: IQPair
    resonator_freq: float
    readout_pulse_length: int
    # ... add other fields here ...

# 4. Define the Quantum Element (q10)
class QuantumElement(BaseModel):
    qubit: QubitParams
    resonator: ResonatorParams

# 5. The Root Config Object
class SystemConfig(BaseModel):
    q10: QuantumElement

    def save_to_json(self, path):
        with open(path, 'w') as f:
            # model_dump_json creates a JSON string of the current state
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load_from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# --- USAGE ---

if __name__ == "__main__":
    # Load parameters
    params = SystemConfig.load_from_json("params.json")

    # READ: No more dictionary strings! Full Autocomplete support.
    old_length = params.q10.qubit.reset_gate.length
    print(f"Current length: {old_length}")

    # WRITE: Type safe
    params.q10.qubit.reset_gate.length = 80
    
    # VALIDATION: This would crash immediately (Good!)
    # params.q10.qubit.T1 = "not a number" 

    # Save back
    params.save_to_json("params.json")
    print("Saved successfully.")