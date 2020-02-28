import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .common import remove_spines
from itertools import cycle

literature_data = {'citations': ['Nimmerjahn et al., 2004',
                                      'Grosche et al., 2013',
                                      'Leahy et al., 2015',
                                      'Olude et al., 2015',
                                      'Appaix et al., 2012', 
                                      'Schreiner et al., 2014',
                                      'Emsley et al., 2006'],
                        'densities': [23000., 15696, 18000, 10700, 12286, 10800, 2666],
                        'errors'   : [1844, 860, 2000, 1750, 1601, 400, 133],
                        'ages' : ['Old', 'Adult', 'Adult', 'Juvenile', 'Juvenile', 'Juvenile', 'Neonate']
                       }


def grouped_barplot(ax, df, cat, subcat, val , err):
    """ Grouped barplot """
    ages = df[subcat].unique()
    x = np.arange(ages.size, dtype=np.int) * 0.8
    for i, age in enumerate(ages):

        dfg = df[df[subcat] == age]

        citations = dfg[cat].values
        densities = dfg[val].values
        errors = dfg[err].values
        colors = dfg['colors'].values

        n_citations = citations.size

        offsets = np.arange(n_citations, dtype=np.float)
        offsets -= offsets.mean()

        for j in range(len(dfg)):

            width = 0.18

            errs = [[0], [errors[j]]]

            ax.bar(x[i] + 0.21 * offsets[j],
                   densities[j],
                   width=width,
                   label=citations[j],
                   yerr=errs,
                   linewidth=1,
                   capsize=10,
                   edgecolor='black',
                   color=colors[j], align='center')

    ax.set_xlabel('Age')
    ax.set_ylabel('Density (Astrocytes / mm^3)')

    ax.set_xlim([-1.5, x[-1] * 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(list(ages))
    ax.set_yticks([5000, 15000, 25000])

    remove_spines(ax, (False, True, False, False))
    #ax.legend(loc='upper left')


def plot_laminar_density_comparison(ax, points, bounding_box):

    volume = 1e-9 * bounding_box.extent.prod()

    saverage = points.shape[0] / volume

    data = {
            'citations': ['Result'],
            'densities': [saverage],
            'errors'   : [0.],
            'ages'     : ['Juvenile']
           }

    for key in literature_data:
        data[key].extend(literature_data[key])

    dataframe = pd.DataFrame.from_dict(data)

    dataframe.sort_values('densities', inplace=True)

    grouped_barplot(ax, dataframe, 'citations', 'ages', 'densities', 'errors')

    remove_spines(ax, (False, True, True, False))
