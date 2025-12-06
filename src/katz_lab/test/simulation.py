from qm_saas import QmSaas
from qm import QuantumMachinesManager
from qm_saas import QOPVersion
from katz_lab.experiments.calibrations.iq_blobs import *
import numpy as np


client = QmSaas(
    email="asaf.solonnikov@mail.huji.ac.il", password="CYbv+v3xVyVvyUhi6o+iEQ=="
)

instance = client.simulator()


with client.simulator() as instance:
    # Use the instance object to simulate QUA programs
    qmm = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )
    # Continue as usual with opening a quantum machine and simulation of a qua program


# %%

options = OptionsIQBlobs()
options.simulate = True

experiment = IQBlobsExperiment(
    qubit="q10",
    options=options,
    qmm=qmm,
)

experiment.run()
