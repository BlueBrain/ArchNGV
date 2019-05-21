""" Cell placement exceptions
"""


class NGVError(Exception):
    """ Generic Cell Placement exeption """


class NotAlignedError(NGVError):
    """ Raised when input datasets area not alligned """
