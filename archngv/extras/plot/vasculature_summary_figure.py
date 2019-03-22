import logging
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from functools import partial
from .common import subsample
from .common import add_layers
from .common import remove_spines
from ..util.graph import properties as graph_prop
from .kernel_density import spatial_kernel_density_plot

sns.set_style("white")

log = logging.getLogger(__name__)


layers = np.array([0.0, 674.68206269999996, 1180.8844627000001, 1363.6375343, 1703.8656135000001, 1847.3347831999999, 2006.3482524000001])

def degrees_plot(ax, graph):

    conts = graph_prop.continuations(graph).sum()
    forks = graph_prop.forks(graph).sum()
    terms = graph_prop.terminations(graph).sum()

    total = conts + forks + terms

    conts_perc = 100. * conts / total
    forks_perc = 100. * forks / total
    terms_perc = 100. * terms / total


    data = {'percentages': (conts_perc, forks_perc, terms_perc),
            'labels': ('Continuations', 'Branches', 'Terminations')}

    sns.barplot(x='percentages',
                y='labels',
                data=pd.DataFrame.from_dict(data),
                estimator=lambda x: x,color='k', alpha=0.6, ax=ax)

    ax.set_xlim([0, 100])


def summary_figure(vasculature, cutoff_radius=6., xz_shape='rectangle', figsize=(20, 15)):

    n_x = 4
    n_y = 4

    f = plt.figure(figsize=figsize)

    bb = vasculature.bounding_box

    x_range, y_range, z_range = bb.ranges.T

    graph = vasculature.point_graph

    points = vasculature.points
    radii  = vasculature.radii
    seg_starts, seg_ends = vasculature.segments
    rad_starts, rad_ends = vasculature.segments_radii
    mean_radii = 0.5 * (rad_starts + rad_ends)

    seg_mask = mean_radii > cutoff_radius
    pnt_mask = radii > cutoff_radius

    #######################################################

    log.info('Cutoff radius: {}'.format(cutoff_radius))

    log.info('Cortical depth wiring plot generating')

    ax1 = plt.subplot2grid((n_x, n_y), (0, 0), rowspan=3)

    cdp = partial(cortical_depth_plot, bounding_box=bb, n_bins=30, xz_shape=xz_shape)

    cdp(ax1, seg_starts, seg_ends, rad_starts, rad_ends, measurement='length', scale=1e3, **{'color': 'k'})
    cdp(ax1, seg_starts[seg_mask], seg_ends[seg_mask], rad_starts[seg_mask], rad_ends[seg_mask], measurement='length', scale=1e3, **{'color': 'r'})
    cdp(ax1, seg_starts[~seg_mask],seg_ends[~seg_mask], rad_starts[~seg_mask], rad_ends[~seg_mask], measurement='length',scale=1e3,  **{'color': 'b'})

    ax1.set_ylabel('Cortical Depth (um)')
    ax1.set_xlabel('Wiring Density (mm / um^3)')

    ax1.set_ylim(y_range)
    remove_spines(ax1, (False, False, False, False))
 
    add_layers(ax1, layers, orientation='horizontal', color='k', alpha=0.6, linestyle='--')

    log.info('Cortical depth wiring plot completed')

    #######################################################

    log.info('Cortical depth area plot generating')

    ax2 = plt.subplot2grid((n_x, n_y), (0, 1), rowspan=3)
    cdp(ax2, seg_starts, seg_ends, rad_starts, rad_ends, measurement='area', **{'color': 'k'})
    cdp(ax2, seg_starts[seg_mask], seg_ends[seg_mask], rad_starts[seg_mask], rad_ends[seg_mask], measurement='area', **{'color': 'r'})
    cdp(ax2, seg_starts[~seg_mask],seg_ends[~seg_mask], rad_starts[~seg_mask], rad_ends[~seg_mask], measurement='area', **{'color': 'b'})

    ax2.set_ylabel('Cortical Depth (um)')
    ax2.set_xlabel('Area Density um^2 / um^3')

    ax2.set_ylim(y_range)
    remove_spines(ax2, (False, False, False, False))


    add_layers(ax2, layers, orientation='horizontal', color='k', alpha=0.6, linestyle='--')

    log.info('Cortical depth area plot completed')

    #######################################################

    log.info('Kernel Density plot generating')

    ax3 = plt.subplot2grid((n_x, n_y), (0, 2), rowspan=3)
    levels = spatial_kernel_density_plot(ax3, points[pnt_mask], x_range, y_range, 'Reds')
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)

    remove_spines(ax3, (False, False, False, False))

    add_layers(ax3, layers, orientation='horizontal', color='k', alpha=0.6, linestyle='--')

    log.info('Kernel Density plot completed')

    #######################################################

    log.info('Kernel Density plot generating')

    ax4 = plt.subplot2grid((n_x, n_y), (0, 3), rowspan=3)
    spatial_kernel_density_plot(ax4, points[~pnt_mask], x_range, y_range, 'Blues', levels=levels)
    ax4.set_xlim(x_range)
    ax4.set_ylim(y_range)

    remove_spines(ax4, (False, False, False, False))

    add_layers(ax4, layers, orientation='horizontal', color='k', alpha=0.6, linestyle='--')

    log.info('Kernel Density plot completed')

    #######################################################

    log.info('Radii histogram  generating')


    ax5 = plt.subplot2grid((n_x, n_y), (3, 0))

    ax5_nbins = 50

    ax5.hist(mean_radii[seg_mask], normed=True, bins=ax5_nbins, color='r')
    ax5.hist(mean_radii[~seg_mask], normed=True, bins=ax5_nbins, color='b')

    ax5.set_xlabel('Radii (um)')
    ax5.set_ylabel('Density')
    ax5.set_xlim([0., 20.])
    remove_spines(ax5, (False, False, False, False))

    log.info('Radii histogram  completed')

    #######################################################

    log.info('Conneted components histogram  generating')

    ax6 = plt.subplot2grid((n_x, n_y), (3, 1))
    components, frequencies, labels = graph_prop.connected_components(graph)
    ax6.hist(np.sort(frequencies[1:]), bins=50)    

    remove_spines(ax6, (False, False, False, False))

    log.info('Conneted components histogram  completed')

    #######################################################

    log.info('Degrees plot generating')

    ax7 = plt.subplot2grid((n_x, n_y), (3, 2), colspan=2)
    degrees_plot(ax7, graph)

    remove_spines(ax7, (False, False, True, False))

    log.info('Degrees plot completed')

    plt.show()
