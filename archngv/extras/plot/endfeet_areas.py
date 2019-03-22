import pickle
import logging
import numpy as np
import seaborn as sns
import pandas as pd


L = logging.getLogger(__name__)


def astrocyte_endfeet_areas(path):

    with open(path, 'r') as fhandler:
        face_idx, areas = pickle.load(fhandler)

    return face_idx, areas


def endfeet_areas_iterator(paths):
    it = map(astrocyte_endfeet_areas, paths)
    return (areas for _, areas in it)


def circuit_endfeet_total_area_iterator(paths):
    return (areas.sum() for areas in endfeet_areas_iterator(paths))


def endfeet_area_percentages(paths, mesh_total_area):

    it = circuit_endfeet_total_area_iterator(paths)
    return np.fromiter(it, dtype=np.float) / mesh_total_area


def plot_endfeet_area_distribution(axis, circuit, plt_options):

    areas = next(endfeet_areas_iterator(circuits))

    axis.hist(areas, **plt_options)


def plot_endfeet_total_area_mean_percentage(axis, circuits, plt_options, mesh_total_area=None):

    percentages = endfeet_area_percentages(circuits, mesh_total_area)

    mean_percentage = np.mean(percentages, axis=0)
    sdev_percentage = np.std(percentages, axis=0)

    L.info('Mean Total Area Percentage: {} +- {}'.format(mean_percentage, sdev_percentage))
    axis.bar(0, mean_percentage, yerr=sdev_percentage, **plt_options)


def plot_endfeet_total_area_percentage(axis, circuits, plt_options, mesh_total_area=None):

    percentages = endfeet_area_percentages(circuits, mesh_total_area)

    axis.bar(np.arange(len(percentages)), percentages, **plt_options)


def total_area_strategy_comparison(experiments, plt_options, mesh_total_area=None):

    data = {'mean_percentages': [],
            'sdev_percentages': [],
            'reachout_strategy': [],
            'max_endfeet': []}

    for circuit in experiments:

        percentages = []

        strategies = []
        max_endfeet = []

        strategies = [repeat.endfeet_reachout_strategy for repeat in circuit]
        max_endfeet = [repeat.endfeet_max_threshold for repeat in circuit]

        assert all([x == strategies[0] for x in strategies])
        assert all([x == max_endfeet[0] for x in  max_endfeet])

        percentages = 100. * endfeet_area_percentages(circuit, mesh_total_area)

        data['mean_percentages'] = np.mean(percentages)
        data['sdev_percentages'] = np.std(percentages)
        data['reachout_strategy'] = strategies[0]
        data['max_endfeet'] = max_endfeet[0]

    df = pd.DataFrame(data)

    return df

    g = sns.factorplot(x='max_endfeet',
                       y='percentages',
                       hue='reachout_strategy',
                       data=df,
                       aspect=1.5, size=4)

    g.set_xlabels('Max Number of Endfeet')
    g.set_ylabels('Vasculature Coverage %')

    return g
