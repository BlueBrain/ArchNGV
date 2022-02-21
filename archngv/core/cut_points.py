""" Data structure for cut point representation
"""

import json

import numpy


class CutPointData:
    """Cut plane data for cut astrocytes"""

    @classmethod
    def load_file(cls, file_path):
        """Load cut plane data file"""
        with open(file_path, "r") as fp:
            cp_dict = json.load(fp)

        morph_name = list(cp_dict.keys())[0]

        cp_dict = cp_dict[morph_name]

        cut_leaves = numpy.asarray(cp_dict["cut_leaves"], dtype=numpy.float)

        orientation, offset = cp_dict["cut_plane"]

        if orientation == "X":

            cut_plane = numpy.array([offset, 0.0, 0.0])

        elif orientation == "Y":

            cut_plane = numpy.array([0.0, offset, 0.0])

        elif orientation == "Z":

            cut_plane = numpy.array([0.0, 0.0, offset])

        else:

            raise TypeError("Unknown Orientation Axis {}".format(orientation))

        details = cp_dict["details"]

        return cls(cut_leaves, cut_plane, details)

    def __init__(self, cut_leaves_coordinates, cut_plane, details, cell=None):

        self.cell = cell
        self.details = details

        self.cut_plane = cut_plane

        self.cut_leaves_coordinates = cut_leaves_coordinates
