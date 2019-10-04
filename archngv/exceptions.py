""" Cell placement exceptions
"""


class NGVError(Exception):
    """ Generic Cell Placement exeption """


class NotAlignedError(NGVError):
    """ Raised when input datasets area not alligned """


class NeuriteNotCreatedError(NGVError):
    """ Raised when a neurite type has failed to be synthesized """
