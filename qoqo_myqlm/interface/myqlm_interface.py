"""Define the MyQLM interface for qoqo operations."""
# Copyright © 2019-2021 HQS Quantum Simulations GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
from qoqo import Circuit
from typing import (
    cast,
    List,
    Any
)
import qat.lang.AQASM as qlm


def myqlm_call_circuit(
        circuit: Circuit,
        number_qubits: int,
        **kwargs) -> qlm.Program:
    """Translate the qoqo circuit into MyQLM ouput

    The qoqo_myqlm interface iterates through the qoqo circuit and translates each qoqo operation
    to MyQLM output (strings).

    Args:
        circuit: The qoqo circuit that is translated
        number_qubits: Number of qubits in the quantum register
        **kwargs: Additional keyword arguments

    Returns:
        qlm.Program: translated circuit
    """
    myqlm_program = qlm.Program()
    qureg = myqlm_program.qalloc(number_qubits)

    for op in circuit:
        if 'PragmaActiveReset' in op.tags():
            myqlm_program.reset(op.involved_qubits)
        else:
            instructions = myqlm_call_operation(op, qureg)
            if instructions is not None:
                myqlm_program.apply(*instructions)

    myqlm_circuit = myqlm_program.to_circ()

    return myqlm_circuit


def myqlm_call_operation(
        operation: Any,
        qureg: qlm.Program.qalloc) -> List:
    """Translate a qoqo operation to MyQLM text

    Args:
        operation: The qoqo operation that is translated
        qureg: The quantum register pyquest_cffi operates on

    Returns:
        List: arguments to be used in the "apply" function

    Raises:
        RuntimeError: Operation not in MyQLM backend
    """
    op = cast(List, None)
    tags = operation.tags()
    if 'RotateZ' in tags:
        op = [qlm.RZ(operation.theta().float()), qureg[operation.qubit()]]
    elif 'RotateX' in tags:
        op = [qlm.RX(operation.theta().float()), qureg[operation.qubit()]]
    elif 'RotateY' in tags:
        op = [qlm.RY(operation.theta().float()), qureg[operation.qubit()]]
    elif 'CNOT' in tags:
        op = [qlm.CNOT, qureg[operation.control()], qureg[operation.target()]]
    elif 'Hadamard' in tags:
        op = [qlm.H, qureg[operation.qubit()]]
    elif 'PauliX' in tags:
        op = [qlm.X, qureg[operation.qubit()]]
    elif 'PauliY' in tags:
        op = [qlm.Y, qureg[operation.qubit()]]
    elif 'PauliZ' in tags:
        op = [qlm.Z, qureg[operation.qubit()]]
    elif 'SGate' in tags:
        op = [qlm.S, qureg[operation.qubit()]]
    elif 'TGate' in tags:
        op = [qlm.T, qureg[operation.qubit()]]
    elif 'ControlledPauliZ' in tags:
        op = [qlm.CSIGN, qureg[operation.control()], qureg[operation.target()]]
    elif 'ControlledPauliY' in tags:
        op = [qlm.Y.ctrl(), qureg[operation.control()], qureg[operation.target()]]
    elif 'SWAP' in tags:
        op = [qlm.SWAP, qureg[operation.control()], qureg[operation.target()]]
    elif 'ISwap' in tags:
        op = [qlm.ISWAP, qureg[operation.control()], qureg[operation.target()]]
    elif 'SingleQubitGate' in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate("Gate", [], arity=1, matrix_generator=matrix)
        op = [gate(), qureg[operation.qubit()]]
    elif 'SingleQubitGateOperation' in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate("Gate", [], arity=1, matrix_generator=matrix)
        op = [gate(), qureg[operation.qubit()]]
    elif 'TwoQubitGateOperation' in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate("Gate", [], arity=2, matrix_generator=matrix)
        op = [gate(), qureg[operation.control()], qureg[operation.target()]]
    elif 'MeasureQubit' in tags:
        pass
    elif 'Definition' in tags:
        pass
    else:
        raise RuntimeError('Operation not in MyQLM backend')

    return op
