# Copyright © 2019-2024 HQS Quantum Simulations GmbH. All Rights Reserved.
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
import numpy as np


def generate_VariableMSXX_matrix(theta: float):
    """Generate the VariableMSXX unitary matrix
    from https://hqsquantumsimulations.github.io/qoqo_examples/gate_operations/two_qubit_gates.html#variablesmsxx
    """

    cos_component = np.cos(theta / 2)
    sin_component = -1j * np.sin(theta / 2)
    U = np.array(
        [
            [cos_component, 0, 0, sin_component],
            [0, cos_component, sin_component, 0],
            [0, sin_component, cos_component, 0],
            [sin_component, 0, 0, cos_component],
        ]
    )

    return U
