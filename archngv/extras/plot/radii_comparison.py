import numpy as np
import pylab as plt
from .common import remove_spines

literature_data = {'value': [5.99, 4.36, 4.85, 5.9, 6.46, 6.05],
                    'citation': ['Puschmann et al., 2014',
                                 'Kali et al., NA',
                                 'Bindocci et al., 2017',
                                 'Lee et al., 2016',
                                 'Guo et al., 2016',
                                 'Bagheri et al., 2013']}


def plot_radii_comparison(ax, radii):

    n_vals = len(literature_data['value'])

    sim_mean = np.mean(radii)
    sim_sdev  = np.std(radii)

    radius_mean = np.mean(literature_data['value'])
    radius_sdev = np.std(literature_data['value'])

    pos = np.arange(n_vals)

    ax.scatter(pos, literature_data['value'],  alpha=0.6, marker='o')
    ax.hlines(radius_mean, 0., 10., color='r', lw=1, label='Literature average')
    ax.hlines(sim_mean, 0., 10.,  color='b',  lw=1, label='Simulation average')
    ax.text(n_vals, radius_mean + 0.5, "{:.2f} +- {:.2f} um".format(radius_mean, radius_sdev), color='r')
    ax.text(n_vals, sim_mean - 1., "{:.2f} +- {:.2f} um".format(sim_mean, sim_sdev), color='b')

    ax.set_ylim([0, 15])

    ax.set_yticks([0., 5., 10.])
    #ax.set_xlabel('Citations')
    ax.set_ylabel('Soma Radius (um)')

    ax.set_xticks(pos)
    ax.set_xticklabels(literature_data['citation'], rotation=45, ha='right')
    ax.legend()

    remove_spines(ax, (False, True, True, False))
