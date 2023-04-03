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
from typing import cast, List, Any
import qat.lang.AQASM as qlm


def myqlm_call_circuit(
    circuit: Circuit, number_qubits: int, noise_mode_all_qubits: bool = False, **kwargs
) -> qlm.Program:
    """Translate the qoqo circuit into MyQLM ouput

    The qoqo_myqlm interface iterates through the qoqo circuit and translates each qoqo operation
    to MyQLM output (strings).

    Args:
        circuit: The qoqo circuit that is translated
        number_qubits: Number of qubits in the quantum register
        noise_mode_all_qubits: boolean to indicate whether to apply noise to all qubits or only to
                                 active qubits
        **kwargs: Additional keyword arguments

    Returns:
        qlm.Program: translated circuit
    """
    myqlm_program = qlm.Program()
    qureg = myqlm_program.qalloc(number_qubits)
    for op in circuit:
        if "PragmaActiveReset" in op.tags():
            myqlm_program.reset(op.involved_qubits)
        elif "PragmaLoop" in op.tags():
            number_of_repetitions = max(0, int(op.repetitions().value))
            for _ in range(number_of_repetitions):
                for op_loop in op.circuit():
                    instructions = myqlm_call_operation(op_loop, qureg)
                    if instructions is not None:
                        myqlm_program.apply(*instructions)
                        if noise_mode_all_qubits:
                            apply_I_on_inactive_qubits(
                                number_qubits, myqlm_program, qureg, instructions
                            )
        else:
            instructions = myqlm_call_operation(op, qureg)
            if instructions is not None:
                myqlm_program.apply(*instructions)
                if noise_mode_all_qubits:
                    apply_I_on_inactive_qubits(
                        number_qubits, myqlm_program, qureg, instructions
                    )

    myqlm_circuit = myqlm_program.to_circ()
    return myqlm_circuit


def apply_I_on_inactive_qubits(
    number_qubits: int,
    myqlm_program: qlm.program.Program,
    qureg: qlm.bits.QRegister,
    instructions: List,
) -> None:
    """Applies an I gate to all inactive qubits in a quantum circuit.

    Args:
        number_qubits: The total number of qubits in the circuit.
        myqlm_program : The QLM program to which to add the gate operations.
        qureg : The quantum register to which to apply the gate operations.
        instructions : A list of instructions specifying the active qubits.
    """
    active_qubits = [int(qb.to_dict()["data"]) for qb in instructions[1:]]
    for qubit in range(number_qubits):
        if qubit not in active_qubits:
            myqlm_program.apply(qlm.I, qureg[qubit])


def myqlm_call_operation(operation: Any, qureg: qlm.Program.qalloc) -> List:  # noqa
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
    if "RotateZ" in tags:
        op = [qlm.RZ(operation.theta().float()), qureg[operation.qubit()]]
    elif "RotateX" in tags:
        op = [qlm.RX(operation.theta().float()), qureg[operation.qubit()]]
    elif "RotateY" in tags:
        op = [qlm.RY(operation.theta().float()), qureg[operation.qubit()]]
    elif "CNOT" in tags:
        op = [qlm.CNOT, qureg[operation.control()], qureg[operation.target()]]
    elif "Hadamard" in tags:
        op = [qlm.H, qureg[operation.qubit()]]
    elif "PauliX" in tags:
        op = [qlm.X, qureg[operation.qubit()]]
    elif "PauliY" in tags:
        op = [qlm.Y, qureg[operation.qubit()]]
    elif "PauliZ" in tags:
        op = [qlm.Z, qureg[operation.qubit()]]
    elif "SGate" in tags:
        op = [qlm.S, qureg[operation.qubit()]]
    elif "TGate" in tags:
        op = [qlm.T, qureg[operation.qubit()]]
    elif "ControlledPauliZ" in tags:
        op = [qlm.CSIGN, qureg[operation.control()], qureg[operation.target()]]
    elif "ControlledPauliY" in tags:
        op = [qlm.Y.ctrl(), qureg[operation.control()], qureg[operation.target()]]
    elif "SWAP" in tags:
        op = [qlm.SWAP, qureg[operation.control()], qureg[operation.target()]]
    elif "ISwap" in tags:
        op = [qlm.ISWAP, qureg[operation.control()], qureg[operation.target()]]
    elif "SingleQubitGate" in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate(tags[-1], [], arity=1, matrix_generator=lambda: matrix)
        op = [gate(), qureg[operation.qubit()]]
    elif "SingleQubitGateOperation" in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate(tags[-1], [], arity=1, matrix_generator=lambda: matrix)
        op = [gate(), qureg[operation.qubit()]]
    elif "TwoQubitGateOperation" in tags:
        matrix = operation.unitary_matrix()
        gate = qlm.AbstractGate(tags[-1], [], arity=2, matrix_generator=lambda: matrix)
        op = [gate(), qureg[operation.control()], qureg[operation.target()]]
    elif "MeasureQubit" in tags:
        pass
    elif "Definition" in tags:
        pass
    elif "PragmaRepeatedMeasurement" in tags:
        pass
    elif "PragmaSetNumberOfMeasurements" in tags:
        pass
    elif "PragmaStartDecompositionBlock" in tags:
        pass
    elif "PragmaGlobalPhase" in tags:
        pass
    elif "PragmaStopDecompositionBlock" in tags:
        pass
    elif "PragmaStopParallelBlock" in tags:
        pass
    else:
        raise RuntimeError(f"Operation not in MyQLM backend tags={tags}")

    return op
