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
from qoqo_myqlm import (
    MyQLMBackend
)
from typing import Any, cast
import math
from hqsbase.calculator import (
    CalculatorFloat,
    Calculator
)
from hqsbase.qonfig import Qonfig


# def test_myqlm_backend():
#     """Testing the MyQLM functionality with the MyQLM backend"""
#     circuit = Circuit()
#     circuit += ops.Definition(name='ro', vartype='bit', length=2, is_output=True)
#     circuit += ops.RotateZ(qubit=0, theta=0)
#     circuit += ops.PauliX(qubit=1)
#     circuit += ops.MeasureQubit(qubit=0)
#     circuit += ops.MeasureQubit(qubit=1)

#     backend = MyQLMBackend(circuit=circuit,
#                            number_qubits=2,
#                            number_measurements=5)

#     results = backend.run()['ro'].register
#     npt.assert_equal(results, [np.array([0., 1.])] * 5)

#     config = backend.to_qonfig()
#     json = config.to_json()
#     config2 = Qonfig.from_json(json)
#     backend2 = config2.to_instance()


if __name__ == '__main__':
    pytest.main(sys.argv)
