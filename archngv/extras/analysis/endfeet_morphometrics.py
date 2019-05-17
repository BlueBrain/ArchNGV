import numpy as np


def total_length(endfoot_section):

    vecs = endfoot_section[1:] - endfoot_section[0:-1]

    return np.linalg.norm(vecs, axis=1).sum()
