from qm_saas import QmSaas

client = QmSaas(
    email="asaf.solonnikov@mail.huji.ac.il", password="CYbv+v3xVyVvyUhi6o+iEQ=="
)


with client.simulator(QmSaas.latest_version()) as instance:
    # Use the instance object to simulate QUA programs
    qmm = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )

    # Continue as usual with opening a quantum machine and simulation of a qua program
