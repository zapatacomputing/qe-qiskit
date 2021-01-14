import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="qe-qiskit",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Integrations for deploying qiskit on Orquestra Quantum Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/qe-qiskit",
    packages=[
        "qeqiskit",
        "qeqiskit.backend",
        "qeqiskit.noise",
        "qeqiskit.optimizer",
        "qeqiskit.simulator",
    ],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "qiskit==0.23.2",
        "qiskit-ibmq-provider==0.11.1",
        "pyquil==2.17.0",
        "numpy>=1.18.1",
        "z-quantum-core",
    ],
)
