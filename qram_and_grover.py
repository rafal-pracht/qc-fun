#
# (C) Copyright Rafał Pracht 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license t http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import math as math


from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute


def to_bin(i, n):
    """Convert a integer to binary with given length."""
    return bin(i)[2:].rjust(n, '0')


def set_integer_value(circuit, addressreg, i, n):
    """Function set the integer value on given register."""
    for index, val in enumerate(to_bin(i, n)[::-1]):
        if val == '0':
            circuit.x(addressreg[index])


def inversion_about_average(circuit, register, ancilla):
    """Grover diffuse operator."""
    circuit.h(register)
    circuit.x(register)
    circuit.h(register[-1])
    circuit.mct(register[:-1], register[-1], ancilla)
    circuit.h(register[-1])
    circuit.x(register)
    circuit.h(register)


def create_oracle(ndata, datareg, oraclereg, fn_create_circuit):
    """Oracle function, it check if the binary number is as expected."""
    oracle = fn_create_circuit()
    for i in range(ndata - 1):
        oracle.cx(datareg[i], oraclereg[i])
        oracle.cx(datareg[i+1], oraclereg[i])
    return oracle


def create_qram(data, naddress, datareg, addressreg, ancillareg, fn_create_circuit):
    """It store the array data in QRAM."""
    qram = fn_create_circuit()
    for i, d in enumerate(data):
        set_integer_value(qram, addressreg, i, naddress)
        for index, val in enumerate(d[::-1]):
            if val == '1':
                qram.mcx(addressreg, datareg[index], ancillareg)
        set_integer_value(qram, addressreg, i, naddress)

    return qram


def optimal_grover_iteration(qubit, M):
    N = qubit ** 2

    theta = np.arcsin(np.sqrt(M / N))
    return round(np.pi / (theta * 4) - 0.5)


def find_solution(data, expected_solution_number, ancilla_n=1):
    """
    Function find the indices of the inputs where two adjacent bits will always have different values.
    :param data: the array of integers
    :param expected_solution_number: the expected number of solutions.
           "You can assume that the input always contains at least two numbers that have alternating bit strings"
    :param ancilla_n: the number of ancilla used by circuit.
    :return: The quantum computer output.

    In this case the output should be: 1/sqrt(2) * (|01> + |11>), as the correct indices are 1 and 3.

    Example:
    Function will design a quantum circuit that considers as input the following vector of integers numbers:
        [1,5,7,10]
    returns a quantum state which is a superposition of indices of the target solution, obtaining in the output
    the indices of the inputs where two adjacent bits will always have different values.
    In this case the output should be: 1/sqrt(2) * (|01> + |11>), as the correct indices are 1 and 3.
        1 = 0001
        5 = 0101
        7 = 0111
        10 = 1010
    The method to follow for this task is to start from an array of integers as input, pass them to a binary
    representation and find those integers whose binary representation is such that two adjacent bits are different.
    Once the function has found those integers, it must output a superposition of states where each state is a binary
    representation of the indices of those integers.
    """
    n_data = max([len(s) for s in [bin(b)[2:] for b in data]])
    binary_data = [to_bin(b, n_data) for b in data]
    org_n_address = math.ceil(math.log(len(data), 2))

    # calculate the grover iteration number
    # this is important, because in the example with 2 good solution on space 2^2=4 grover search will fail,
    # because the optimal number of iteration is zero :)
    k = optimal_grover_iteration(org_n_address, expected_solution_number)
    if k == 0:
        n_address = org_n_address + 1
        k = optimal_grover_iteration(n_address, expected_solution_number)
    else:
        n_address = org_n_address

    # Crate registers
    dataReg = QuantumRegister(n_data)
    addressReg = QuantumRegister(n_address)
    oracleReg = QuantumRegister(n_data-1)
    ancillaReg = QuantumRegister(ancilla_n)
    addressCreg = ClassicalRegister(org_n_address)

    create_circuit = lambda: QuantumCircuit(dataReg, addressReg, oracleReg, ancillaReg, addressCreg)

    qc = create_circuit()
    qc.h(addressReg)
    qram = create_qram(binary_data, n_address, dataReg, addressReg, ancillaReg, create_circuit)
    oracle = create_oracle(n_data, dataReg, oracleReg, create_circuit)

    for _ in range(k):
        # oracle
        qc = qc.compose(qram)
        qc = qc.compose(oracle)
        ###
        qc.barrier()
        qc.h(oracleReg[-1])
        qc.mcx(oracleReg[:-1], oracleReg[-1], ancillaReg)
        qc.h(oracleReg[-1])
        qc.barrier()
        # oracle inv
        qc = qc.compose(oracle.inverse())
        qc = qc.compose(qram.inverse())

        # dyffuse
        inversion_about_average(qc, addressReg, ancillaReg)

    if n_address == org_n_address:
        qc.measure(addressReg, addressCreg)
    else:
        qc.measure(addressReg[:-1], addressCreg)

    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend).result()
    counts = result.get_counts(qc)

    return counts


########################################################################################################################
#                                   Use cases                                                                          #
########################################################################################################################
# one, solution
c = find_solution([1, 5, 7, 11], 1)
assert set(c.keys()) == {'01'}

# example from task
c = find_solution([1, 5, 7, 10], 2)
assert set(c.keys()) == {'01', '11'}

# The bigger array
c = find_solution([1, 5, 7, 10, 6, 8], 2)
assert set(c.keys()) == {'001', '011'}

c = find_solution([0, 2, 1, 5, 7, 10, 6, 8, 15], 2)
# there is 1024 shots, store only the solution more than 1% (10)
assert set([k for (k, v) in c.items() if v > 10]) == {'0011', '0101'}
