

class TransmonParameters:
    def __init__(self):

class Transmon:
    def __init__(self, qubit_name, parameters: TransmonParameters):
        self.qubit_name = qubit_name
        self.parameters: TransmonParameters = parameters

class QPU:
    def __init__(self, qubits: list[Transmon]):
        self.qubits = qubits

        