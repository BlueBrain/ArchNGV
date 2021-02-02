"""Add endoplasmic reticulum to astrocyte
morphologies
"""
import numpy as np


def add_endoplasmic_reticulum_to_morphology(morphology):
    """Adds endoplasmic reticulum placeholder values.

    Morphologies need to be populated with ER. However, the algos for synthesizing the ER
    is not yet in place. Thus, in order to enable the testing for the neurodamus implementations,
     we add placeholder values for the time being.

    Args:
        morhology (morphio.mut.Morphology): morphio mutable morphology
    """
    n_sections = len(morphology.sections)

    er = morphology.endoplasmic_reticulum
    er.section_indices = np.arange(n_sections, dtype=np.int)
    er.surface_areas = np.full(n_sections, fill_value=0.1, dtype=np.float32)
    er.volumes = np.full(n_sections, fill_value=0.2, dtype=np.float32)
    er.filament_counts = np.ones(n_sections, dtype=np.int)
