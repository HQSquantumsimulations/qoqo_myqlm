"""Testing MyQLM backend"""

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
from qoqo_myqlm import MyQLMBackend


def test_myqlm_backend():
    """Testing the MyQLM backend simple run"""
    circuit = Circuit()
    circuit += ops.DefinitionBit(name="ro", length=2, is_output=True)
    circuit += ops.RotateZ(qubit=0, theta=0)
    circuit += ops.PauliX(qubit=1)
    circuit += ops.MeasureQubit(qubit=0, readout="ro", readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout="ro", readout_index=1)

    backend = MyQLMBackend(number_qubits=2, number_measurements=5)

    (bit_dict, float_dict, complex_dict) = backend.run_circuit(circuit)
    npt.assert_equal(float_dict, dict())
    npt.assert_equal(complex_dict, dict())
    npt.assert_equal(bit_dict["ro"], [np.array([0.0, 1.0])] * 5)


@pytest.mark.parametrize(
    "thetas, outcome",
    [  # 2*pi rotation with equal thetas
        (
            (np.pi, np.pi),
            [[False, False]],
        ),
        # pi rotation with 2 different thetas
        (
            (np.pi / 3, 2 * np.pi / 3),
            [[True, True]],
        ),
        # two random rotations that gives two states as outcomes
        (
            (np.pi / 2, np.pi / 3),
            [[False, False], [True, True]],
        ),
    ],
)
def test_myqlm_backend_with_VariableMSXX(thetas, outcome):
    """Testing a run with just 2 VariableMSXX gates,
    with different theta value.
    """
    circuit = Circuit()
    circuit += ops.DefinitionBit(name="ro", length=2, is_output=True)
    circuit += ops.VariableMSXX(0, 1, theta=thetas[0])
    circuit += ops.VariableMSXX(0, 1, theta=thetas[1])
    circuit += ops.MeasureQubit(qubit=0, readout="ro", readout_index=0)
    circuit += ops.MeasureQubit(qubit=1, readout="ro", readout_index=1)

    backend = MyQLMBackend(number_qubits=2, number_measurements=0)
    (bit_dict, _, _) = backend.run_circuit(circuit)
    npt.assert_equal(bit_dict["ro"], outcome)


if __name__ == "__main__":
    pytest.main(sys.argv)
