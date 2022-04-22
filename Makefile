################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv ${VENV} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./z-quantum-core && \
		${VENV}/bin/python3 -m pip install -e '.[develop]'

build-system-deps:
	$(PYTHON) -m pip install setuptools wheel "setuptools_scm>=6.0"
