"""MyQLM interface for qoqo.

Translates qoqo operations and circuits to MyQLM operations via the interface,
and Creates a MyQLM file with MyQLMBackend.

.. autosummary::
    :toctree: generated/

    MyQLMBackend
    myqlm_call_circuit
    myqlm_call_operation

"""

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
from qoqo_myqlm.backend import (
    MyQLMBackend,
)
from qoqo_myqlm.interface import (
    generate_VariableMSXX_matrix,
    myqlm_call_circuit,
    myqlm_call_operation,
)

__all__ = (
    "MyQLMBackend",
    "generate_VariableMSXX_matrix",
    "myqlm_call_circuit",
    "myqlm_call_operation",
)
