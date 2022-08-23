"""Backend producing MyQLM"""
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
    Optional,
    Dict,
    List,
    Tuple,
    cast,
    Any
)
from qoqo_myqlm.interface import myqlm_call_circuit
import numpy as np
import warnings
import qat
from qat.qpus import get_default_qpu


class MyQLMBackend(object):
    r"""Backend to qoqo that produces MyQLM output which can be imported.

    This backend takes a qoqo circuit to be run on a certain device and returns a MyQLM file
    containing the translated circuit. The circuit itself is translated using the qoqo_myqlm
    interface.
    """

    def __init__(self,
                 number_qubits: int = 1,
                 number_measurements: int = 1,
                 device: Optional[Any] = None,  # noqa
                 job_type: str = "SAMPLE",
                 observable: Optional[np.ndarray] = None,
                 qpu: Any = None) -> None:  # noqa
        """Initialize MyQLM Backend

        Args:
            number_qubits: The number of qubits to use
            number_measurements: The number of measurement repetitions. If set to 0:
                - simulator: tries to output all the possible final states (with probabilities)
                - quantum processor: uses the largest amount of shots authorised by the hardware
            device: The device specification
            job_type: MyQLM job type to run:
                - SAMPLE (default): measures Z on all qubits
                - OBS: measures a specific observable, defined by a matrix on all qubits
            observable: if "OBS" is selected as the job type, this is the matrix of
                        the observable to measure.
            qpu: QPU machine to use (quantum processor or simulator) with relevant keywords

        Raises:
            TypeError: Job_type specified is neither 'SAMPLE' nor 'OBS'
        """
        self.name = "myqlm"
        self.number_qubits = number_qubits
        self.number_measurements = number_measurements
        self.device = device
        self.job_type = job_type
        if qpu is None:
            qpu = get_default_qpu()
        self.qpu = qpu

        if job_type == "SAMPLE":
            if observable is not None:
                warnings.warn("SAMPLE job type given, ignoring the observable matrix")
            self.observable = None
        elif job_type == "OBS":
            if observable is None:
                warnings.warn(
                    "OBS job_type given without observable matrix, using Z on all qubits")
                observable = np.array([[1, 0], [0, -1]])
            self.observable = observable
        else:
            raise TypeError("Job_type specified is neither 'SAMPLE' nor 'OBS'")

    def run_circuit(self, circuit: Circuit
                    ) -> Tuple[Dict[str, List[List[bool]]],
                               Dict[str, List[List[float]]],
                               Dict[str, List[List[complex]]]]:
        """Turn the circuit into MyQLM and save to file

        Args:
            circuit: The circuit that is run

        Returns:
            Tuple[Dict[str, List[List[bool]]],
                  Dict[str, List[List[float]]],
                  Dict[str, List[List[complex]]]]
        """
        # Initializing the classical registers for calculation and output
        internal_bit_register_dict: Dict[str, List[bool]] = dict()
        internal_float_register_dict: Dict[str, List[float]] = dict()
        internal_complex_register_dict: Dict[str, List[complex]] = dict()

        output_bit_register_dict: Dict[str, List[List[bool]]] = dict()
        output_float_register_dict: Dict[str, List[List[float]]] = dict()
        output_complex_register_dict: Dict[str, List[List[complex]]] = dict()

        for bit_def in circuit.filter_by_tag("DefinitionBit"):
            internal_bit_register_dict[bit_def.name()] = [False for _ in range(bit_def.length())]
            if bit_def.is_output():
                output_bit_register_dict[bit_def.name()] = list()

        for float_def in circuit.filter_by_tag("DefinitionFloat"):
            internal_float_register_dict[float_def.name()] = [
                0.0 for _ in range(float_def.length())]
            if float_def.is_output():
                output_float_register_dict[float_def.name()] = cast(List[List[float]], list())

        for complex_def in circuit.filter_by_tag("DefinitionComplex"):
            internal_complex_register_dict[complex_def.name()] = [
                complex(0.0) for _ in range(complex_def.length())]
            if complex_def.is_output():
                output_complex_register_dict[complex_def.name()] = cast(List[List[complex]], list())

        compiled_circuit = myqlm_call_circuit(circuit, self.number_qubits)

        if self.observable is None:
            job = compiled_circuit.to_job(job_type='SAMPLE',
                                          nbshots=self.number_measurements,
                                          aggregate_data=False)
        else:
            obs = qat.core.Observable(nqbits=self.number_qubits, matrix=self.observable)
            job = compiled_circuit.to_job(job_type='OBS',
                                          nbshots=self.number_measurements,
                                          observable=obs,
                                          aggregate_data=False)

        result = self.qpu.submit(job)
        for sample in result:
            array = [qubit_state for qubit_state in sample.state]
            output_bit_register_dict['ro'].append(array)

        return output_bit_register_dict, output_float_register_dict, output_complex_register_dict

    def run_measurement_registers(self, measurement: Any  # noqa
                                  ) -> Tuple[Dict[str, List[List[bool]]],
                                             Dict[str, List[List[float]]],
                                             Dict[str, List[List[complex]]]]:
        """Run all circuits of a measurement with the PyQuEST backend

        Args:
            measurement: The measurement that is run

        Returns:
            Tuple[Dict[str, List[List[bool]]],
                  Dict[str, List[List[float]]],
                  Dict[str, List[List[complex]]]]
        """
        constant_circuit = measurement.constant_circuit()
        output_bit_register_dict: Dict[str, List[List[bool]]] = dict()
        output_float_register_dict: Dict[str, List[List[float]]] = dict()
        output_complex_register_dict: Dict[str, List[List[complex]]] = dict()
        for circuit in measurement.circuits():
            if constant_circuit is None:
                run_circuit = circuit
            else:
                run_circuit = constant_circuit + circuit

            (tmp_bit_register_dict,
             tmp_float_register_dict,
             tmp_complex_register_dict) = self.run_circuit(
                run_circuit
            )
            output_bit_register_dict.update(tmp_bit_register_dict)
            output_float_register_dict.update(tmp_float_register_dict)
            output_complex_register_dict.update(tmp_complex_register_dict)
        return (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict)

    def run_measurement(self, measurement: Any  # noqa
                        ) -> Optional[Dict[str, float]]:
        """Run a circuit with the PyQuEST backend

        Args:
            measurement: The measurement that is run

        Returns:
            Optional[Dict[str, float]]
        """
        (output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict) = self.run_measurement_registers(measurement)
        return measurement.evaluate(
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict)
