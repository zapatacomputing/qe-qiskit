# qe-qiskit

[![codecov](https://codecov.io/gh/zapatacomputing/qe-qiskit/branch/master/graph/badge.svg?token=G64YYS2IOS)](https://codecov.io/gh/zapatacomputing/qe-qiskit)

An Orquestra Quantum Engine Resource for Qiskit

## Overview

Contained in this repository is an Orquestra quantum engine resource for using qiskit. For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).


## QiskitSimulator

Included in this resource is the `QiskitSimulator` which implements the `zquantum.core.interfaces.QuantumSimulator` interface. 

It can be imported with:
```
from qeqiskit.simulator import QiskitSimulator
```

## QiskitBackend

Included in this resource is the `QiskitBackend` which implements the `zquantum.core.interfaces.QuantumBackend` interface.

It can be imported with:
```
from qeqiskit.backend import QiskitBackend
```
