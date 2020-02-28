import os
import logging

import numpy as np

from archngv import NGVConfig, NGVCircuit

L = logging.getLogger(__name__)


class DataPlots:

    def __init__(self, circuit_properties, plots_config):
        self.prop = circuit_properties
        self.plot = plots_config


class PaperFigure(object):

    def __init__(self, config_path, output_directory):
        self._config = NGVConfig.from_file(config_path)
        self._circuit = NGVCircuit(self._config)

        self._prefix = None

        self._extension = 'pdf'

    def _save_figure(self, fig, name):

        assert self._prefix is not None

        filename = '{}.{}'.format(name, self._extension)
        path = os.path.join(self._output_directory, filename)

        fig.savefig(path, transparent=True)
        L.info('Figure saved at: {}'.format(path))

    


LAYERS = {'bins': np.array([0.0,
                            674.68206269999996,
                            1180.8844627000001,
                            1363.6375343,
                            1703.8656135000001,
                            1847.3347831999999,
                            2006.3482524000001]),
          'labels': ['L6', 'L5', 'L4', 'L3', 'L2', 'L1'],
          'centers': np.array([337.34103135,
                               927.7832627,
                               1272.2609985,
                               1533.7515739,
                               1775.60019835,
                               1926.8415178])
}


class Layers:

    def __init__(self):
        self._layers = LAYERS

    def __len__(self):
        return len(self.labels)

    @property
    def bins(self):
        return self._layers['bins']

    @property
    def centers(self):
        bs = self.bins
        return 0.5 * (bs[1:] + bs[:-1])

    @property
    def labels(self):
        return self._layers['labels']

    def thicknesses(self):
        return np.diff(self.bins)


FIGURE_PROPS = {'font.size'          : 15,
         'axes.labelsize'     : 18,
         'axes.titlesize'     : 20,
         'legend.fontsize'    : 10,
         'xtick.labelsize'    : 15,
         'ytick.labelsize'    : 15,
         'xtick.direction'    : 'out',
         'ytick.direction'    : 'out',
         'xtick.major.size'   : 6,
         'xtick.major.width'  : 2,
         'axes.spines.right'  : False,
         'axes.spines.top'    : False,
         'figure.facecolor'   : 'white',
         'savefig.dpi'        : 300}
