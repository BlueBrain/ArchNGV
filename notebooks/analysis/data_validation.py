import os
import json
import logging
import pathlib

from copy import deepcopy
from cached_property import cached_property

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import pylab as plt
import seaborn.apionly as sns

from archngv import NGVConfig, NGVCircuit
from archngv.extras.plot.common import remove_spines
from archngv.extras.plot.average_density_comparison import grouped_barplot
from archngv.extras.analysis.nearest_neighbor import nearest_neighbor_distances
from archngv.utils.statistics import truncated_normal

from common import LAYERS
from common import DataPlots
from plots_globals import PlotsConfig
from plots_globals import stepfilled_histogram

from plots_globals import RESULT_COLOR
from plots_globals import DUAL_COMPARISON_COLOR
from circuit_properties import CircuitProperties


L = logging.getLogger('__name__')


class SomataPlots(DataPlots):

    def density_profile(self, ax, density_validation_path):
        from archngv.extras.plot.densities import plot_spatial_distribution_histogram

        voxel_density = self.prop.voxel_density
        astrocyte_positions = self.prop.astrocyte_positions
        bbox = self.prop.bbox

        ref_densities = {
            'voxel_data': voxel_density,
            'color': DUAL_COMPARISON_COLOR,
            'label': 'Appaix et al., 2012'
        }

        point_population_dict = {
            'Result': {
                'points': astrocyte_positions,
                'color': RESULT_COLOR,
                'edgecolor': 'white'
            }
        }

        plot_spatial_distribution_histogram(ax, ref_densities, bbox, point_population_dict)

        ax.set_ylabel("Cortical Depth (um)")
        ax.set_xlabel("Astrocyte Density\n(astrocytes / cubic mm)")

        ax.set_xticks([0, 20000])

        ax.set_xlim([0., 20000])
        ax.set_ylim([600, 2120])

        ax.legend(loc=2)

        l_bins = LAYERS['bins']
        aligned_bins = l_bins - l_bins.max() + voxel_density.offset[1] + \
                                               voxel_density.voxel_dimensions[0] * \
                                               voxel_density.shape[1]

        aligned_centers = 0.5 * (aligned_bins[1:] + aligned_bins[:-1])

        ax.set_yticks(aligned_centers[1:])
        ax.set_yticklabels(LAYERS['labels'][1:])
        ax.xaxis.set_tick_params(direction='out', which='bottom')


        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(False)

        ax.xaxis.tick_top()
        ax.xaxis.label_position = 'top'
        plt.tight_layout()

    def density_comparison(self, ax, literature_data):

        volume = 1e-9 * self.prop.bbox.extent.prod()
        astrocyte_positions = self.prop.astrocyte_positions

        saverage = len(astrocyte_positions) / volume

        data = {
                'citations': ['Result'],
                'densities': [saverage],
                'errors'   : [0.],
                'ages'     : ['Juvenile'],
               }

        for key in literature_data:
            data[key].extend(literature_data[key])

        data['colors'] = [RESULT_COLOR] + [DUAL_COMPARISON_COLOR] * len(literature_data['citations'])

        dataframe = pd.DataFrame.from_dict(data)
        dataframe.sort_values('densities', inplace=True)

        grouped_barplot(ax, dataframe, 'citations', 'ages', 'densities', 'errors')

        remove_spines(ax, (False, True, True, False))

        ax.set_ylabel("Astrocytes / mm$^3$")

    def radius_distribution_comparison(self, ax, literature_radii):

        astrocyte_radii = self.prop.astrocyte_radii

        literature_radii = deepcopy(literature_radii)
        n_entries = len(literature_radii['citation']) + 1

        colors = [DUAL_COMPARISON_COLOR for _ in range(n_entries - 1)]
        colors.append(RESULT_COLOR)

        mean_val = np.mean(astrocyte_radii)
        sdev_val = np.std(astrocyte_radii)

        literature_radii['value'].append(mean_val)
        literature_radii['citation'].append('Result')
        literature_radii['sdev'].append(sdev_val)

        column_positions = np.arange(n_entries, dtype=np.int)
        column_widths = literature_radii['value']

        devs = literature_radii['sdev']
        #ax.errorbar(x_values, y_values, x_sdevs, y_sdevs, fmt='_', ecolor=colors)
        ax.barh(column_positions, column_widths,
                color=colors, edgecolor='black', capsize=5, xerr=[[0] * len(devs), devs])

        ax.set_yticks(range(n_entries))
        ax.set_yticklabels(literature_radii['citation'])

        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        ax.set_xlim([4, 6.7])

        ax.set_xlabel('Radius ($\mathrm{\mu}$m)')

        plt.tight_layout()

    def radius_distribution_histogram(self, ax):

        astrocyte_radii = self.prop.astrocyte_radii

        stepfilled_histogram(ax, values=astrocyte_radii, color=RESULT_COLOR)

        ax.set_xlabel('Radius ($\mathrm{\mu}$m)')
        ax.set_ylabel('Astrocytes')

        ax.yaxis.set_major_locator(plt.LinearLocator(3))
        ax.xaxis.set_major_locator(plt.LinearLocator(3))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))
        plt.tight_layout()
    
    def volume_distribution(self, ax, literature_volumes):

        volumes = self.prop.astrocyte_somata_volumes
        n_volumes = len(volumes)

        dict_data = {'volume': volumes[:].tolist(),
                     'source': ['Simulation'] * n_volumes}

        for mean, sdev, source in zip(literature_volumes['mean'], literature_volumes['sdev'], literature_volumes['source']):

            exp_volumes = stats.norm(loc=mean, scale=sdev).rvs(size=n_volumes)

            dict_data['volume'] += exp_volumes.tolist()
            dict_data['source'] += [source] * n_volumes

        data = pd.DataFrame(dict_data)

        sns.violinplot(x='source', y='volume', data=data, ax=ax)
        #ax.hist(volumes, bins=20, normed=True)
        remove_spines(ax, (False, True, True, False))

        #ax.set_xlim([0, 2000])

    def astrocyte_nearest_neighbor_distribution(self, ax, validation_data):

        validation_data = pd.DataFrame(validation_data).copy()
        astrocyte_positions = self.prop.astrocyte_positions
        distx = nearest_neighbor_distances(astrocyte_positions, astrocyte_positions)

        xmin = 10.
        xmax = 50.

        h, _, _ = ax.hist(distx, bins=30, histtype='stepfilled', density=True,
                          color=RESULT_COLOR, lw=1, edgecolor='black', label='Result')

        for _, row in validation_data.iterrows():

            t = np.linspace(xmin, xmax, 1000)

            f_dstr = stats.norm(loc=row['mean'], scale=row['sdev']).pdf

            ax.plot(t, f_dstr(t), label=row['citation'], lw=3, color=DUAL_COMPARISON_COLOR)

        ax.set_ylim([0, 1.1 * h.max()])
        ax.set_xlim([xmin, xmax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([10, 20, 30, 40, 50])

        ax.set_xlabel('Nearest Distance  ($\mathrm{\mu}$m)')

        ax.legend(loc=1)
        plt.tight_layout()


class MicrodomainPlots(DataPlots):
    
    def volume_distribution(self, ax):

        microdomain_volumes = self.prop.microdomain_volumes
        overlapping_volumes = self.prop.overlapping_microdomain_volumes

        hist_kws = {'histtype': 'stepfilled'}
        kde_kws = {'linewidth': 4}

        sns.distplot(microdomain_volumes / 1e4, ax=ax,
                    color=DUAL_COMPARISON_COLOR, hist_kws=hist_kws, hist=False, kde_kws=kde_kws)
        sns.distplot(overlapping_volumes / 1e4, ax=ax,
                    color=RESULT_COLOR, hist_kws=hist_kws, hist=False, kde_kws=kde_kws)

        ax.set_xlim([0, 25])
        ax.spines['left'].set_visible(False)

        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.NullLocator())

        ax.set_xlabel('Volume ' + r"($\times \mathrm{10^4}$ $\mathrm{\mu m^3}$)")
        plt.tight_layout()

    def volume_distribution_violin(self, ax):

        norm_volumes = self.prop.microdomain_volumes
        over_volumes = self.prop.overlapping_microdomain_volumes

        my_pal = {"Tight": DUAL_COMPARISON_COLOR, "Overlapping": RESULT_COLOR}

        types = ['Tight'] * len(norm_volumes) + ['Overlapping'] * len(over_volumes)

        volumes_df = pd.DataFrame({'type': types,
                                   'volume': np.hstack((norm_volumes, over_volumes))})

        sns.violinplot(x="type", y="volume", data=volumes_df, ax=ax, palette=my_pal, saturation=1)

        ax.set_xlabel('')
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_ylabel('Volume ' + r"($\times \mathrm{10^4}$ $\mathrm{\mu m^3}$)")

        plt.tight_layout()
    
    def volume_comparison(self, ax, literature_data):

        vols = self.prop.overlapping_microdomain_volumes

        comp_df = pd.DataFrame(literature_data)

        simulation_entry = {'mean_volume': [np.mean(vols)],
                            'sdev_volume': [np.std(vols)],
                            'citation': ['Result'],
                            'species': ['Rat'],
                            'staining_method': ['In Silico']}

        comp_df = pd.concat([comp_df, pd.DataFrame(simulation_entry)])
        comp_df.sort_values('mean_volume', inplace=True, ascending=False)

        colors = [DUAL_COMPARISON_COLOR] * len(comp_df)
        colors[1] = RESULT_COLOR


        sns.stripplot(data=comp_df, size=12, orient="h",
                      edgecolor="w", y='citation', x='mean_volume',
                      ax=ax, palette=colors)

        ax.yaxis.grid(True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('')
        ax.set_xticks([0, 50000, 100000])
        ax.set_xlabel('Mean Volume ($\mathrm{\mu m ^3}$)')

        for i, t in enumerate(ax.yaxis.get_ticklabels()):
            if i == 1:
                t.set_color(colors[i])


        x_coords = []
        y_coords = []
        for point_pair in ax.collections:
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)

        colors = [DUAL_COMPARISON_COLOR] * len(x_coords)
        colors[1] = RESULT_COLOR

        # Calculate the type of error to plot as the error bars
        # Make sure the order is the same as the points were looped over
        errors = comp_df['sdev_volume'].values

        ax.errorbar(x_coords, y_coords, xerr=errors,
                    ecolor=colors, fmt=' ', elinewidth=2)

        plt.tight_layout()


class EndfeetPlots(DataPlots):

    def areas_histogram(self, ax, literature_data):

        areas = self.prop.endfeet_surface_meshes

        ax.hist(areas[~np.isclose(areas, 0.0)], bins=50, color=RESULT_COLOR, histtype='stepfilled', edgecolor='k')
        ax.set_xlabel('Areas ($\mu$m$^2$)')
        ax.set_ylabel('Number of Endfeet')
        ax.set_xlim([0, 800])

    def areas_cumulative_validation(self, ax, literature_data):

        def areas_ecdf(vals):
            n = float(len(areas))
            ys = np.arange(1, n + 1) / n
            xs = np.sort(areas)
            return xs[::100], ys[::100]

        areas = self.prop.endfeet_surface_meshes
        unreduced_areas = self.prop.endfeet_areas_unreduced

        d = truncated_normal(literature_data['mean'], literature_data['sdev'], 0., 1000.)

        areas = areas[~np.isclose(areas, 0.0)]
        unreduced_areas = unreduced_areas[~np.isclose(unreduced_areas, 0.0)]

        r = truncated_normal(areas.mean(), areas.std(), 0., 1000.)
        t = np.linspace(0., areas.max(), 100)

        xs, ys = areas_ecdf(areas)
        ax.plot(xs, d.cdf(xs), color=DUAL_COMPARISON_COLOR, linewidth=4, label='Cali et al., 2019')
        ax.plot(xs, ys, color=RESULT_COLOR, linewidth=4, linestyle='--', label='Result with reduction')

        xs, ys = areas_ecdf(unreduced_areas)
        ax.plot(xs, ys, color=RESULT_COLOR, linewidth=4, linestyle='--', label='Result wout reduction')

        ax.legend(frameon=False, fontsize=20)

    def path_lengths_histogram(self, ax, literature_data):

        path_lengths = self.prop.endfeet_path_lengths
        stepfilled_histogram(ax, values=path_lengths, color=RESULT_COLOR, bins=30)
        ax.set_xlabel('Path Length ($\mu$m)')

    def path_lengths_bar_comparison(self, ax, literature_data):

        path_lengths = self.prop.endfeet_path_lengths

        path_mean = path_lengths.mean()
        path_sdev = path_lengths.std()

        bio_mean = literature_data['mean'][0]
        bio_sdev = literature_data['sdev'][0]

        x = [0, 1, 2, 3]
        colors = ['white', RESULT_COLOR, DUAL_COMPARISON_COLOR, 'white']

        ax.bar(x, [0, path_mean, bio_mean, 0], width=0.4, color=colors, edgecolor='k',
               yerr=[[0, 0, 0, 0], [0, path_sdev, bio_sdev, 0]], linewidth=2, capsize=10, align='center')


        ax.set_xticks(x)
        ax.set_xticklabels(['', 'Result', literature_data['citation'][0], ''])

        ax.set_ylabel('Path length')

    def area_coverage_pie(self, ax, literature_data):

        endfeet_areas = self.prop.endfeet_surface_meshes

        center = plt.Circle((0, 0), 0.7, color='white')

        #vasculature_area = self.vasculature.area() * 0.6
        endfeet_total_area = np.sum(endfeet_areas)
        mesh_area = 22444090.01175495
        perc = endfeet_total_area / mesh_area

        sizes = [100. * perc, 100. * (1. - perc)]
        explode = [0.0, 0.1]

        labels = ['Endfeet area', 'Remaining area']

        ax.pie(sizes, explode=explode,
               colors=[RESULT_COLOR, DUAL_COMPARISON_COLOR],
               autopct='{:1.0f}%'.format, shadow=True, labels=labels)

    def volume_area_scatter(self, ax, literature_data):

        endfeet_areas = self.prop.endfeet_surface_meshes
        endfeet_thicknesses = self.prop.endfeet_thicknesses
        endfeet_volumes = endfeet_areas * endfeet_thicknesses

        mask = ~np.isclose(endfeet_volumes, 0.0)

        ax.scatter(endfeet_volumes[::100], endfeet_areas[::100], s=0.5, color=RESULT_COLOR)

        ax.set_ylabel('Areas ($\mu$m$^2$)')
        ax.set_xlabel('Volumes ($\mu$m$^3$)')

        a, b = np.polyfit(endfeet_volumes, endfeet_areas, 1)

        t = np.linspace(0., endfeet_volumes.max(), 4)

        ax.plot(t, a * t + b, color=RESULT_COLOR,
                label='y = {:.2f}x + {:.2f}'.format(a, b), linewidth=2)

        bio_a, bio_b = literature_data['ab']

        ax.plot(t, bio_a * t + bio_b, color=DUAL_COMPARISON_COLOR,
                label='y = {:.2f}x + {:.2f}'.format(bio_a, bio_b),
                linestyle='dotted', linewidth=3,)

        ax.legend(frameon=False)


class MorphologyPlots(DataPlots):

    def primary_processes_per_astrocyte_bar(self, ax, literature_data):

        primary_per_astrocytes = self.prop.primary_processes_per_astrocyte

        prim_mean = np.mean(primary_per_astrocytes)
        prim_sdev = np.std(primary_per_astrocytes)

        values = [prim_mean]
        errors = [[0], [prim_sdev]]
        pos = [0]

        colors = [RESULT_COLOR]
        labels = ['Result']

        citation_means = literature_data['mean']
        citation_sdevs = literature_data['sdev']
        citations = literature_data['citation']


        for i, (mean, sdev, citation) in enumerate(zip(citation_means, citation_sdevs, citations)):
            values.append(mean)
            errors[0].append(0)
            errors[1].append(sdev)
            pos.append(i + 1)
            colors.append(DUAL_COMPARISON_COLOR)
            labels.append(citation)

        ax.bar(pos, values, yerr=errors, color=colors, width=0.3, 
                edgecolor='k', linewidth=2, capsize=10, align='center')

        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel('Primary processes per astrocyte')
        ax.set_ylabel('Number of astrocytes')

    def perivascular_processes_per_astrocyte_bar(self, ax, literature_data):

        periv_per_astro = self.prop.number_of_endfeet_per_astrocyte

        periv_mean = np.mean(periv_per_astro)
        periv_sdev = np.std(periv_per_astro)


        values = [periv_mean]
        errors = [[0], [periv_sdev]]
        pos = [0]
        colors = [RESULT_COLOR]
        labels = ['Result']

        citation_means = literature_data['mean']
        citation_sdevs = literature_data['sdev']
        citations = literature_data['citation']

        for i, (mean, sdev, citation) in enumerate(zip(citation_means, citation_sdevs, citations)):
            values.append(mean)
            errors[0].append(0)
            errors[1].append(sdev)
            pos.append(i + 1)
            colors.append(DUAL_COMPARISON_COLOR)
            labels.append(citation)

        ax.bar(pos, values, yerr=errors, color=colors, width=0.4, 
                edgecolor='k', linewidth=2, capsize=10, align='center')

        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=90)

        ax.set_xlabel('Number of endfeet per astrocyte')
        ax.set_ylabel('Number of astrocytes')
        #ax.hist(n_endfeets_per_astrocyte,
        #        color=RESULT_COLOR, bins=6, edgecolor='black', histtype='stepfilled')
 


class DataValidationPlots:

    def __init__(self, config_path, output_directory):

        self.prop = CircuitProperties(config_path)
        self.plot = PlotsConfig(output_directory)

        self.somata = SomataPlots(self.prop, self.plot)
        self.microdomains = MicrodomainPlots(self.prop, self.plot)
        self.endfeet = EndfeetPlots(self.prop, self.plot)
        self.morphologies = MorphologyPlots(self.prop, self.plot)

        self.save_figure = self.plot.save_figure

'''

from archngv.extras.analysis.pair_correlation import pairCorrelationFunction_3D
from archngv.extras.analysis.pair_correlation import RipleyKFunction
from archngv.extras.analysis.pair_correlation import RipleyLFunction
from archngv.extras.plot.common import smooth_convolve

class DataValidationPlots(PlotsConfig, CircuitProperties):

    def __init__(self, config_path, output_directory):

        PlotsConfig.__init__(self, output_directory)
        CircuitProperties.__init__(self, config_path)
        self.layers = LAYERS

    def _get_preprocessed_positions(self):

        x, y, z = self.astrocyte_positions.T

        xoff, yoff, zoff = self.bbox.offset.T

        # remove the offset
        x -= xoff
        y -= yoff
        z -= zoff

        cx = x.mean()
        cy = y.mean()
        cz = z.mean()

        box_window_size = min(self.bbox.extent)

        hwidth = 0.5 * box_window_size

        # extract cubic window
        mask = (x >= cx - hwidth) & (x <= cx + hwidth) & \
               (y >= cy - hwidth) & (y <= cy + hwidth) & \
               (z >= cz - hwidth) & (z <= cz + hwidth)

        return x[mask], y[mask], z[mask], box_window_size

    def pair_correlation(self, ax):

        x, y, z, box_window_size = self._get_preprocessed_positions()

        rMax =  0.1 * box_window_size

        dr = 1.0

        L.info("dr: {}, rMax: {}".format(dr, rMax))

        g_average, radii, _ = pairCorrelationFunction_3D(x, y, z, box_window_size, rMax, dr)
        g_average = smooth_convolve(g_average, window_len=8)

        x_u = np.random.uniform(box_window_size, size=len(x))
        y_u = np.random.uniform(box_window_size, size=len(x))
        z_u = np.random.uniform(box_window_size, size=len(x))

        # uniform pattern
        g_average_u, radii_u, _ = pairCorrelationFunction_3D(x_u, y_u, z_u, box_window_size, rMax, dr)
        g_average_u = smooth_convolve(g_average_u, window_len=8)

        ax.set_xlabel('Shell Radius (um)')
        ax.set_ylabel('Pair Correlation')


        #ax.set_yticks([0., 1., 2., 3.])
        #ax.set_yticklabels([0, 1., 2. , 3.])
        ax.set_xticks([0., 30., 60, 90.])

        remove_spines(ax, (False, True, True, False))

        ax.plot(radii, g_average, color='darkred', alpha=0.9, linewidth=2.5, label='Simulation Pattern')
        ax.plot(radii_u, g_average_u, color='b', alpha=0.9, linewidth=2.5, label='Uniform Pattern')

        ax.set_xlim([0., 90.])
        ax.set_ylim([0., np.max([np.max(g_average), np.max(g_average_u)])])

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1f}'.format(x)))

        ax.legend()
        plt.tight_layout()


    def ripleys_L_function(self, ax):

        x, y, z, box_window_size = self._get_preprocessed_positions()

        rMax =  0.1 * box_window_size

        dr = 1.0

        L.info("dr: {}, rMax: {}".format(dr, rMax))


        l_average, radii, _ = RipleyLFunction(x, y, z, box_window_size, rMax, dr)
        l_average = smooth_convolve(l_average, window_len=8)


        x_u = np.random.uniform(box_window_size, size=len(x))
        y_u = np.random.uniform(box_window_size, size=len(x))
        z_u = np.random.uniform(box_window_size, size=len(x))

        # uniform pattern
        l_average_u, radii_u, _ = RipleyLFunction(x_u, y_u, z_u, box_window_size, rMax, dr)
        l_average_u = smooth_convolve(l_average_u, window_len=8)

        ax.set_xlabel('Shell Radius(um)')
        ax.set_ylabel('L Function')

        #ax.set_yticks([0., 1., 2., 3.])
        #ax.set_yticklabels([0, 1., 2. , 3.])
        ax.set_xticks([0., 30., 60., 90.])

        remove_spines(ax, (False, True, True, False))

        ax.plot(radii, l_average, color='darkred', alpha=0.9, linewidth=2.5, label='Simulation Pattern')
        ax.plot(radii_u, l_average_u, color='b', alpha=0.9, linewidth=2.5, label='Uniform Pattern')

        ax.legend()

        plt.tight_layout()
 '''
