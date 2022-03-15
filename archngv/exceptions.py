""" Cell placement exceptions
"""


class NGVError(Exception):
    """Generic Cell Placement exception"""


class NotAlignedError(NGVError):
    """Raised when input datasets area not aligned"""


class NeuriteNotCreatedError(NGVError):
    """Raised when a neurite type has failed to be synthesized"""
