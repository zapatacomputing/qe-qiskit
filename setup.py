################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import site
import sys
import warnings

import setuptools

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    warnings.warn("Unable to import extras")
    extras = {}

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="qe-qiskit",
    use_scm_version=True,
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Integrations for deploying qiskit on Orquestra Quantum Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/qe-qiskit",
    packages=setuptools.find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "qiskit>=0.28, <0.34",
        "qiskit-ibmq-provider~=0.15",
        "symengine~=0.7",
        "numpy~=1.0",
        "z-quantum-core",
    ],
    extras_require=extras,
    include_package_data=True,
    zip_safe=False,
)
