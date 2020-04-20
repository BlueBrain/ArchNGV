"""NGV constant container for the different ngv objects"""

from bluepysnap.sonata_constants import ConstContainer
# redirection of useful snap containers for neuronal circuits. This allows
# from archngv.ngv_constants import Cell, Synapse
from bluepysnap.bbp import Cell, Synapse  # pylint: disable=unused-import


class Astrocyte(ConstContainer):
    """Astrocyte property names."""

    X = "x"
    Y = "y"
    Z = "z"

    MORPHOLOGY = "morphology"

    MTYPE = "mtype"
    RADIUS = "radius"
    MODEL_TEMPLATE = "model_template"


class Population(ConstContainer):
    """NGV population names."""

    ASTROCYTES = "astrocytes"
    NEUROGLIAL = "neuroglial"
    GLIALGLIAL = "glialglial"
    GLIOVASCULAR = "gliovascular"
    VASCULATURE = "vasculature"
    ENDFEET = "endfeetome"
    MICRODOMAINS = "microdomains"
