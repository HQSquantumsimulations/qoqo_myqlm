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
from qoqo import operations as ops
from qoqo import Circuit
from typing import (
    Optional,
    Dict,
    cast,
    List,
    Any
)
from hqsbase.calculator import (
    Calculator,
    CalculatorFloat,
)
import numpy as np
import qat.lang.AQASM as qlm


# Create look-up tables

_MyQLM_ARGUMENT_NAME_DICTS: Dict[str, Dict[str, CalculatorFloat]] = dict()
_MyQLM_DEFAULT_EXPONENT: float = cast(float, None)
_MyQLM_NAME: Dict[str, Any] = dict()

_MyQLM_ARGUMENT_NAME_DICTS['Hadamard'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['Hadamard'] = qlm.H
_MyQLM_ARGUMENT_NAME_DICTS['PauliX'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['PauliX'] = qlm.X
_MyQLM_ARGUMENT_NAME_DICTS['PauliY'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['PauliY'] = qlm.Y
_MyQLM_ARGUMENT_NAME_DICTS['PauliZ'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['PauliZ'] = qlm.Z
_MyQLM_ARGUMENT_NAME_DICTS['SGate'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['SGate'] = qlm.S
_MyQLM_ARGUMENT_NAME_DICTS['TGate'] = {'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['TGate'] = qlm.T
_MyQLM_ARGUMENT_NAME_DICTS['RotateZ'] = {'qubit': ('qubits', 'qubit'),
                                         'theta': ('parameters', 'theta')}
_MyQLM_NAME['RotateZ'] = qlm.RZ
_MyQLM_ARGUMENT_NAME_DICTS['RotateX'] = {'qubit': ('qubits', 'qubit'),
                                         'theta': ('parameters', 'theta')}
_MyQLM_NAME['RotateX'] = qlm.RX
_MyQLM_ARGUMENT_NAME_DICTS['RotateY'] = {'qubit': ('qubits', 'qubit'),
                                         'theta': ('parameters', 'theta')}
_MyQLM_NAME['RotateY'] = qlm.RY
_MyQLM_ARGUMENT_NAME_DICTS['CNOT'] = {'control': ('qubits', 'control'),
                                      'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['CNOT'] = qlm.CNOT
_MyQLM_ARGUMENT_NAME_DICTS['ControlledPauliZ'] = {'control': ('qubits', 'control'),
                                                  'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['ControlledPauliZ'] = qlm.CSIGN
_MyQLM_ARGUMENT_NAME_DICTS['ControlledPauliY'] = {'control': ('qubits', 'control'),
                                                  'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['ControlledPauliY'] = qlm.Y.ctrl()
_MyQLM_ARGUMENT_NAME_DICTS['SWAP'] = {'control': ('qubits', 'control'),
                                      'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['SWAP'] = qlm.SWAP
_MyQLM_ARGUMENT_NAME_DICTS['ISwap'] = {'control': ('qubits', 'control'),
                                       'qubit': ('qubits', 'qubit')}
_MyQLM_NAME['ISwap'] = qlm.ISWAP


# Defining the actual call

def myqlm_call_circuit(
        circuit: Circuit,
        number_qubits: int,
        calculator: Optional[Calculator] = None,
        **kwargs) -> qlm.Program:
    """Translate the qoqo circuit into MyQLM ouput

    The qoqo_myqlm interface iterates through the qoqo circuit and translates each qoqo operation
    to MyQLM output (strings).

    Args:
        circuit: The qoqo circuit that is translated
        number_qubits: Number of qubits in the quantum register
        calculator: The HQSBase Calculator used to replace symbolic parameters
        **kwargs: Additional keyword arguments

    Returns:
        qlm.Program: translated circuit
    """
    myqlm_program = qlm.Program()
    qureg = myqlm_program.qalloc(number_qubits)

    for op in circuit:
        if 'PragmaActiveReset' in op._operation_tags:
            myqlm_program.reset(op.involved_qubits)
        else:
            instructions = myqlm_call_operation(op,
                                                qureg,
                                                calculator,
                                                **kwargs)
            if instructions is not None:
                myqlm_program.apply(*instructions)

    myqlm_circuit = myqlm_program.to_circ()

    return myqlm_circuit


def myqlm_call_operation(
        operation: ops.Operation,
        qureg: qlm.Program.qalloc,
        calculator: Optional[Calculator] = None,
        **kwargs) -> List:
    """Translate a qoqo operation to MyQLM text

    Args:
        operation: The qoqo operation that is translated
        qureg: The quantum register pyquest_cffi operates on
        calculator: The HQSBase Calculator used to replace symbolic parameters
        **kwargs: Additional keyword arguments

    Returns:
        List: arguments to be used in the "apply" function

    Raises:
        OperationNotInBackendError: Operation not in MyQLM backend
    """
    op = cast(List, None)
    tags = operation._operation_tags
    if 'RotateZ' in tags:
        op = _execute_GateOperation(
            operation, 'RotateZ', qureg, calculator, **kwargs)
    elif 'RotateX' in tags:
        op = _execute_GateOperation(
            operation, 'RotateX', qureg, calculator, **kwargs)
    elif 'RotateY' in tags:
        op = _execute_GateOperation(
            operation, 'RotateY', qureg, calculator, **kwargs)
    elif 'CNOT' in tags:
        op = _execute_GateOperation(
            operation, 'CNOT', qureg, calculator, **kwargs)
    elif 'Hadamard' in tags:
        op = _execute_GateOperation(
            operation, 'Hadamard', qureg, calculator, **kwargs)
    elif 'PauliX' in tags:
        op = _execute_GateOperation(
            operation, 'PauliX', qureg, calculator, **kwargs)
    elif 'PauliY' in tags:
        op = _execute_GateOperation(
            operation, 'PauliY', qureg, calculator, **kwargs)
    elif 'PauliZ' in tags:
        op = _execute_GateOperation(
            operation, 'PauliZ', qureg, calculator, **kwargs)
    elif 'SGate' in tags:
        op = _execute_GateOperation(
            operation, 'SGate', qureg, calculator, **kwargs)
    elif 'TGate' in tags:
        op = _execute_GateOperation(
            operation, 'TGate', qureg, calculator, **kwargs)
    elif 'ControlledPauliZ' in tags:
        op = _execute_GateOperation(
            operation, 'ControlledPauliZ', qureg, calculator, **kwargs)
    elif 'ControlledPauliY' in tags:
        op = _execute_GateOperation(
            operation, 'ControlledPauliY', qureg, calculator, **kwargs)
    elif 'SWAP' in tags:
        op = _execute_GateOperation(
            operation, 'SWAP', qureg, calculator, **kwargs)
    elif 'ISwap' in tags:
        op = _execute_GateOperation(
            operation, 'ISwap', qureg, calculator, **kwargs)
    elif 'SingleQubitGate' in tags:
        op = _execute_AbstractGateOperation(
            operation, qureg, calculator, **kwargs)
    elif 'SingleQubitGateOperation' in tags:
        op = _execute_AbstractGateOperation(
            operation, qureg, calculator, **kwargs)
    elif 'TwoQubitGateOperation' in tags:
        op = _execute_AbstractGateOperation(
            operation, qureg, calculator, **kwargs)
    elif 'MeasureQubit' in tags:
        pass
    elif 'Definition' in tags:
        pass
    else:
        raise ops.OperationNotInBackendError('Operation not in MyQLM backend')

    return op


def _execute_AbstractGateOperation(
        operation: ops.Operation,
        qureg: qlm.Program.qalloc,
        calculator: Optional[Calculator] = None,
        **kwargs) -> List:
    operation = cast(ops.SingleQubitGate, operation)
    parameter_dict: Dict[str, CalculatorFloat] = dict()
    if calculator is not None:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = (calculator.parse_get(sarg.value))
    else:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = sarg.value

    def generate_matrix() -> np.ndarray:
        return operation.unitary_matrix_from_parameters(**parameter_dict)
    qubits = operation.involved_qubits
    gate = qlm.AbstractGate("Gate", [], arity=len(qubits), matrix_generator=generate_matrix)

    instructions = [gate()]
    for qubit in qubits:
        instructions.append(qureg[qubit])

    return instructions


def _execute_GateOperation(
        operation: ops.Operation,
        tag: str,
        qureg: qlm.Program.qalloc,
        calculator: Optional[Calculator] = None,
        **kwargs) -> List:
    operation = cast(ops.GateOperation, operation)
    qubits: List[str] = list()
    qkwargs: List[float] = list()
    gate = _MyQLM_NAME[tag]

    parameter_dict: Dict[str, CalculatorFloat] = dict()
    if calculator is not None:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = calculator.parse_get(sarg.value)
    else:
        for key, sarg in operation._ordered_parameter_dict.items():
            parameter_dict[key] = sarg.value

    for key in _MyQLM_ARGUMENT_NAME_DICTS[tag].keys():
        dict_name, dict_key = _MyQLM_ARGUMENT_NAME_DICTS[tag][key]
        if dict_name == 'qubits':
            arg = operation._ordered_qubits_dict[dict_key]
            qubits.append(arg)
        elif dict_name == 'parameters':
            parg = parameter_dict[dict_key]
            qkwargs.append(parg)

    if not qkwargs:
        instructions = [gate]
    else:
        instructions = [gate(*qkwargs)]

    for qubit in qubits:
        instructions.append(qureg[qubit])

    return instructions
