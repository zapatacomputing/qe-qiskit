import setuptools

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
    setup_requires=["setuptools_scm~=6.0"],
    install_requires=[
        "qiskit~=0.28",
        "qiskit-ibmq-provider~=0.15",
        "symengine~=0.7",
        "numpy~=1.0",
        "z-quantum-core",
    ],
)
