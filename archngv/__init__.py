# SPDX-License-Identifier: Apache-2.0

""" Archngv, the pipeline for the Neuronal - Glial - Vascular architecture
"""
from importlib.metadata import version

__version__ = version(__package__)

from archngv.core.circuit import NGVCircuit
