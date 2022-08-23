"""Test MyQLM interface"""
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
import pytest
import sys
import numpy as np
import numpy.testing as npt
from qoqo import operations as ops
from qoqo import Circuit
from qoqo_myqlm import myqlm_call_circuit, myqlm_call_operation
import qat.lang.AQASM as qlm

qubits = qlm.Program().qalloc(2)


@pytest.mark.parametrize("gate", [
    (ops.RotateX(0, -np.pi), [qlm.RX(-np.pi), qubits[0]]),
    (ops.RotateY(0, -np.pi), [qlm.RY(-np.pi), qubits[0]]),
    (ops.RotateZ(0, -np.pi), [qlm.RZ(-np.pi), qubits[0]]),
    (ops.CNOT(1, 0), [qlm.CNOT, qubits[1], qubits[0]]),
    (ops.Hadamard(0), [qlm.H, qubits[0]]),
    (ops.PauliX(0), [qlm.X, qubits[0]]),
    (ops.PauliY(0), [qlm.Y, qubits[0]]),
    (ops.PauliZ(0), [qlm.Z, qubits[0]]),
    (ops.SGate(0), [qlm.S, qubits[0]]),
    (ops.TGate(0), [qlm.T, qubits[0]]),
    (ops.ControlledPauliY(1, 0), [qlm.Y.ctrl(), qubits[1], qubits[0]]),
    (ops.ControlledPauliZ(1, 0), [qlm.CSIGN, qubits[1], qubits[0]]),
    (ops.MeasureQubit(0, 'ro', 0), None),
    (ops.DefinitionBit('ro', 1, False), None)
])
def test_gate_translation(gate):
    """Test gate operations with MyQLM interface"""
    myqlm_operation = myqlm_call_operation(operation=gate[0],
                                           qureg=qubits)

    assert myqlm_operation == gate[1]


def test_circuit_translation():
    """Test translation of a full circuit with MyQLM interface"""
    circuit = Circuit()
    circuit += ops.DefinitionBit('ro', is_output=True, length=2)
    circuit += ops.Hadamard(qubit=0)
    circuit += ops.RotateX(qubit=1, theta=np.pi/2)
    circuit += ops.CNOT(control=0, target=1)
    circuit += ops.MeasureQubit(qubit=0, readout='ro', readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout='ro', readout_index=1)

    myqlm_program = qlm.Program()
    qubits = myqlm_program.qalloc(2)
    myqlm_program.apply(qlm.H, qubits[0])
    myqlm_program.apply(qlm.RX(np.pi/2), qubits[1])
    myqlm_program.apply(qlm.CNOT, qubits[0], qubits[1])
    myqlm_circuit = myqlm_program.to_circ()

    translated_circuit = myqlm_call_circuit(circuit=circuit, number_qubits=2)

    for op_trans, op_orig in zip(translated_circuit.iterate_simple(), myqlm_circuit.iterate_simple()):
        assert op_trans == op_orig


if __name__ == '__main__':
    pytest.main(sys.argv)
