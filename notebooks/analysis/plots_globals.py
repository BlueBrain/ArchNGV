import os
import logging
import numpy as np


L = logging.getLogger(__name__)


RESULT_COLOR = '#fe9f64'
DUAL_COMPARISON_COLOR = 'gray'


def stepfilled_histogram(ax, values, color, **kwargs):
    return ax.hist(
        values,
        color=RESULT_COLOR,
        histtype='stepfilled',
        edgecolor='black',
        **kwargs)


class PlotsConfig:

    def __init__(self, output_directory):

        self.figure_extensions = ['pdf', 'png']
        self.output_directory = output_directory

    def save_figure(self, fig, axes, name):
        for ext in self.figure_extensions:

            path = os.path.join(self.output_directory, f'{name}.{ext}')
            fig.savefig(path, transparent=True)
            L.info('Figure saved: {}'.format(path))

            if isinstance(axes, list):
                pass
            elif isinstance(axes, np.ndarray):
                axes = axes.ravel()
            else:
                axes = [axes]

            ticklabels_to_become_visible = []

            for ax in axes:

                for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
                    if ticklabel.get_visible():
                        ticklabels_to_become_visible.append(ticklabel)
                        ticklabel.set_visible(False)

                title = ax.get_title()
                ax.set_title('')

                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                ax.set_xlabel('')
                ax.set_ylabel('')

                try:
                    ax.get_legend().set_visible(False)
                except AttributeError:
                    pass

            path = os.path.join(self.output_directory, f'{name}_no_axes.{ext}')
            fig.savefig(path, transparent=True)


            for ticklabel in ticklabels_to_become_visible:
                ticklabel.set_visible(True)

            for ax in axes:

                for tick in ax.get_xticklabels():
                    tick.set_visible(True)
                for tick in ax.get_yticklabels():
                    tick.set_visible(True)

                ax.set_title(title)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                try:
                    ax.get_legend().set_visible(True)
                except AttributeError:
                    pass

