"""NGV constant container for the different ngv objects"""

# redirection of useful snap containers for neuronal circuits. This allows
# from archngv.ngv_constants import Cell, Synapse
from bluepysnap.bbp import Cell, Synapse  # pylint: disable=unused-import
from bluepysnap.sonata_constants import ConstContainer


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

    NEURONS = "biophysical"
    ASTROCYTES = "astrocytes"
    VASCULATURE = "vasculature"

    NEURONAL = "neuronal"
    GLIALGLIAL = "glialglial"
    NEUROGLIAL = "neuroglial"
    GLIOVASCULAR = "gliovascular"
