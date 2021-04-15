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
from qoqo.backends import (
    BackendBaseClass,
)
from qoqo import Circuit
from hqsbase.calculator import Calculator
from typing import (
    Optional,
    Dict,
    cast,
    Union,
    Any
)
from qoqo.registers import (
    FloatRegister,
    BitRegister,
    ComplexRegister,
    RegisterOutput,
    add_register
)
from qoqo.devices import DeviceBaseClass
from hqsbase.qonfig import Qonfig, empty
from qoqo_myqlm import myqlm_call_circuit
import numpy as np
import warnings
import qat
from qat.qpus import get_default_qpu


class MyQLMBackend(BackendBaseClass):
    r"""Backend to qoqo that produces MyQLM output which can be imported.

    This backend takes a qoqo circuit to be run on a certain device and returns a MyQLM file
    containing the translated circuit. The circuit itself is translated using the qoqo_myqlm
    interface.
    """

    _qonfig_defaults_dict = {
        'circuit': {'doc': 'The circuit that is run',
                    'default': None},
        'number_qubits': {'doc': 'The number of qubits to use',
                          'default': empty},
        'substitution_dict': {'doc': 'Substitution dictionary used to replace symbolic parameters',
                              'default': None},
        'number_measurements': {'doc': 'The number of measurement repetitions',
                                'default': 1},
        'device': {'doc': 'The device specification',
                   'default': None},
        'job_type': {'doc': 'MyQLM job type to run (SAMPLE/OBS)',
                     'default': "SAMPLE"},
        'observable': {'doc': 'if "OBS" is selected as the job type, this is the matrix of '
                              + 'the observable to measure',
                       'default': None},
        'qpu': {'doc': 'QPU machine to use (quantum processor or simulator)',
                'default': None},
    }

    def __init__(self,
                 circuit: Optional[Circuit] = None,
                 number_qubits: int = 1,
                 substitution_dict: Optional[Dict[str, float]] = None,
                 number_measurements: int = 1,
                 device: Optional[DeviceBaseClass] = None,
                 job_type: str = "SAMPLE",
                 observable: Optional[np.ndarray] = None,
                 qpu: Any = None,
                 **kwargs) -> None:
        """Initialize MyQLM Backend

        Args:
            circuit: The circuit that is run
            number_qubits: The number of qubits to use
            substitution_dict: Substitution dictionary used to replace symbolic parameters
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
            kwargs: Additional keyword arguments

        Raises:
            TypeError: Job_type specified is neither 'SAMPLE' nor 'OBS'
        """
        self.name = "myqlm"

        self._circuit = circuit
        self.number_qubits = number_qubits
        self.substitution_dict = substitution_dict
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

        if self.substitution_dict is None:
            self.calculator = None
        else:
            self.calculator = Calculator()
            for name, val in self.substitution_dict.items():
                self.calculator.set(name, val)

        super().__init__(circuit=self._circuit,
                         substitution_dict=self.substitution_dict,
                         device=self.device,
                         number_qubits=number_qubits,
                         **kwargs)

        if circuit is None:
            circuit = Circuit()
        self.compiled_circuit = myqlm_call_circuit(circuit=self._circuit,
                                                   number_qubits=self.number_qubits,
                                                   calculator=self.calculator,
                                                   **kwargs)

    @classmethod
    def from_qonfig(cls,
                    config: Qonfig['MyQLMBackend']
                    ) -> 'MyQLMBackend':
        """Create an Instance from Qonfig

        Args:
            config: Qonfig of class

        Returns:
            MyQLMBackend
        """
        if isinstance(config['circuit'], Qonfig):
            init_circuit = config['circuit'].to_instance()
        else:
            init_circuit = cast(Optional[Circuit], config['circuit'])
        if isinstance(config['device'], Qonfig):
            init_device = config['device'].to_instance()
        else:
            init_device = cast(Optional[DeviceBaseClass], config['device'])
        return cls(circuit=init_circuit,
                   number_qubits=config['number_qubits'],
                   substitution_dict=config['substitution_dict'],
                   number_measurements=config['number_measurements'],
                   device=init_device,
                   job_type=config['job_type'],
                   observable=config['observable'],
                   qpu=get_default_qpu(),
                   )

    def to_qonfig(self) -> 'Qonfig[MyQLMBackend]':
        """Create a Qonfig from Instance

        Returns:
            Qonfig[MyQLMBackend]
        """
        config = Qonfig(self.__class__)
        if self._circuit is not None:
            config['circuit'] = self._circuit.to_qonfig()
        else:
            config['circuit'] = self._circuit
        config['number_qubits'] = self.number_qubits
        config['substitution_dict'] = self.substitution_dict
        config['number_measurements'] = self.number_measurements
        if self.device is not None:
            config['device'] = self.device.to_qonfig()
        else:
            config['device'] = self.device
        config['job_type'] = self.job_type
        config['observable'] = self.observable
        config['qpu'] = str(type(self.qpu))

        return config

    def run(self, **kwargs) -> Union[None, Dict[str, 'RegisterOutput']]:
        """Turn the circuit into MyQLM and save to file

        Args:
            kwargs: Additional keyword arguments

        Returns:
            Union[None, Dict[str, 'RegisterOutput']]
        """
        # Initializing the classical registers for calculation and output
        internal_register_dict: Dict[str, Union[BitRegister,
                                                FloatRegister, ComplexRegister]] = dict()
        output_register_dict: Dict[str, RegisterOutput] = dict()
        for definition in self.circuit._definitions:
            add_register(internal_register_dict, output_register_dict, definition)
        if self.observable is None:
            job = self.compiled_circuit.to_job(job_type='SAMPLE',
                                               nbshots=self.number_measurements,
                                               aggregate_data=False)
        else:
            obs = qat.core.Observable(nqbits=self.number_qubits, matrix=self.observable)
            job = self.compiled_circuit.to_job(job_type='OBS',
                                               nbshots=self.number_measurements,
                                               observable=obs,
                                               aggregate_data=False)

        result = self.qpu.submit(job)
        conv_dict = {False: 0., True: 1.}
        for sample in result:
            array = [conv_dict.get(qubit_state) for qubit_state in sample.state]
            output_register_dict['ro'].register.append(np.array(array))

        return output_register_dict
