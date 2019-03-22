import numpy as np

from tmd.Topology.transformations import tmd_scale



def scale_barcode(ph, target_distance):
    """ Given a target distance, scale the persistence homology
    in order to make sure that the neurite will reach the target
    """
    ph_distance = np.nanmax(ph)

    if target_distance > ph_distance:
        ph = tmd_scale(ph, target_distance / ph_distance)

    return ph
