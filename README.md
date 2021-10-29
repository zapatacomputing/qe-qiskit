# qe-qiskit

[![codecov](https://codecov.io/gh/zapatacomputing/qe-qiskit/branch/main/graph/badge.svg?token=G64YYS2IOS)](https://codecov.io/gh/zapatacomputing/qe-qiskit)

An Orquestra Quantum Engine Resource for Qiskit

## Overview

`qe-qiskit` is a Python module that exposes Qiskit's simulators as a [`z-quantum-core`](https://github.com/zapatacomputing/z-quantum-core/blob/main/src/python/zquantum/core/interfaces/backend.py) `QuantumSimulator`. It can be imported with:

```
from qeqiskit.simulator import QiskitSimulator
```

It also exposes Qiskit's quantum backends as a `QiskitBackend` which implements the `zquantum.core.interfaces.QuantumBackend` interface.

It can be imported with:

```
from qeqiskit.backend import QiskitBackend
```

In addition, it also provides converters that allow switching between `qiskit` circuits and those of `z-quantum-core`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Qiskit integration docs](http://docs.orquestra.io/other-resources/framework-integrations/qiskit/).

For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).
