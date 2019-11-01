import numpy as np
import logging


L = logging.getLogger(__name__)


# TODO: these need to be passed in as parameters
LAYERS = {'bins': np.array([0.0,
                            674.68206269999996,
                            1180.8844627000001,
                            1363.6375343,
                            1703.8656135000001,
                            1847.3347831999999,
                            2006.3482524000001]),
          'labels': ('VI', 'V', 'IV', 'III', 'II', 'I'),
          'int_labels': (6, 5, 4, 3, 2, 1),
          'centers': np.array([337.34103135,
                               927.7832627,
                               1272.2609985,
                               1533.7515739,
                               1775.60019835,
                               1926.8415178])
          }


def find_layer(y_coordinates):
    '''use `LAYERS` to find the layer for the `y_coordinates`'''
    bin_index = np.searchsorted(LAYERS['bins'], y_coordinates)

    bin_index = np.clip(bin_index, 0, 5)

    return LAYERS['int_labels'][bin_index]
