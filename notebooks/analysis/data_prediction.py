
import numpy as np
import pandas as pd

from archngv.extras.plot.average_density_comparison import grouped_barplot

from common import Layers
from common import DataPlots 
from plots_globals import RESULT_COLOR

from plots_globals import PlotsConfig
from circuit_properties import CircuitProperties

# triadic complementary colors
COLOR_TRIPLET = ['#8066fc', '#66fc80', '#fc8066']

# tetradic complementary colors
COLOR_QUADRUPLET = []


def _stacked_barplot_layer(entities, values_per_entity, colors, ax):

    index = pd.Index(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'], name='Layer')

    
    df = pd.DataFrame({name: values for name, values in zip(entities, values_per_entity)},
                      index=index)

    df.plot.barh(stacked=True, ax=ax, color=colors)

    for i, (_, row) in enumerate(df.iterrows()):

        values = row.to_numpy()
        bins = np.r_[0, np.cumsum(values)]

        widths = np.diff(bins)
        centers = bins[:-1] + 0.5 * widths

        
        for val, c in zip(values, centers):
            ax.text(c, i, '{:.1f}'.format(val), color='k',
                    horizontalalignment='center',
                    verticalalignment='center')

class DataPredictionPlots(DataPlots):

    def __init__(self, config_path, output_directory):

        self.prop = CircuitProperties(config_path)
        self.plot = PlotsConfig(output_directory)

        self.plots = Plots(self.prop, self.plot)

        self.save_figure = self.plot.save_figure


def _in_layer(positions, low, high):
    ys = positions[:, 1]
    return (low <= ys) & (ys < high)


class Plots(DataPlots):

    def wiring_across_layers(self, ax, statistics):
        """
        statistics:
            Dataframe with per layer total statistics:
            [entity, layer, length, area, volume]
        """
        df = statistics.copy()

        colors = {
            'neuron': COLOR_TRIPLET[0],
            'astrocyte': COLOR_TRIPLET[1],
            'vasculature': COLOR_TRIPLET[2]
        }

        df['colors'] = [colors[entity] for entity in df.loc[:, 'entity']]
        df['error'] = 0

        grouped_barplot(ax, df, 'entity', 'layer', 'length', 'error')

        ax.set_xlabel('Total Process Length (um)')

        ax.set_yscale('log')
        ax.set_ylabel('Total process length (m)')
        ax.set_xlabel('Layer')
        ax.set_xlim([-0.5, 3.7])


    def volumes_across_layers_polar(self, axes, statistics):

        thetas = [ i * np.pi / 3. for i in range(6) ]
        thetas.append(thetas[0])

        thetas = np.asarray(thetas)

        total_rs = np.zeros(7)

        for i, entity in enumerate(['neuron', 'astrocyte', 'vasculature']):

            mask = statistics.loc[:, 'entity'] == entity

            rs = statistics.loc[mask, 'volume'].to_list()
            rs.append(rs[0])
            
            rs = np.asarray(rs)
            total_rs += rs

            axes[i].fill(thetas, rs, color=COLOR_TRIPLET[i], alpha=0.6)
            axes[i].plot(thetas, rs, '-o', color=COLOR_TRIPLET[i])
            axes[i].set_xticks(thetas)
            axes[i].set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
            #axes[i].set_yticklabels([])

    def density_across_layers(self):

        bbox = self.prop.bbox
        layers = Layers()
        thicknesses = layers.thicknesses()

        astrocyte_positions = self.prop.astrocyte_positions
        neuronal_positions = self.prop.neuronal_positions
        #vascular_positions = self.prop.vasculature_positions

        depth = 0.0
        for layer, thickness in enumerate(thicknesses):

            new_depth = depth - thickness
            mask_astros = _in_layer(astrocyte_positions, low=new_depth, high=depth)
            #mask_vasculature = _in_layer(vasculature_positions, low=new_depth, high=depth)
            depth = 0.

        

    def microdomain_volume_per_layer(self, ax):
    
        layers = Layers()

        bins = layers.bins

        pos = self.prop.astrocyte_positions
        microdomains = self.prop.overlapping_microdomains

        ids = np.digitize(pos[:, 1], bins)

        means = np.zeros(len(bins) - 1)
        sdevs = np.zeros(len(bins) - 1)

        for i in range(len(layers)):

            m_ids = np.where(ids == i)[0]

            vols = np.fromiter((microdomains[index].volume for index in m_ids), dtype=np.float)
           
            if len(vols) > 2:
                means[i] = np.mean(vols)
                sdevs[i] = np.std(vols)

        means = means[1:]
        labels = layers.labels[1:]

        pos = np.arange(len(means))
        ax.plot(pos, means)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)

    def _point_density_plot(self, ax, positions, cmap):

        from archngv.extras.plot.kernel_density import spatial_kernel_density_plot

        bbox = self.prop.bbox
        x_range = bbox.min_point[0], bbox.max_point[0]
        y_range = bbox.min_point[1], bbox.max_point[1]

        spatial_kernel_density_plot(ax, positions, x_range, y_range, cmap=cmap)        

    def spatial_density_astrocytic_somata(self, ax):
        self._point_density_plot(ax, self.prop.astrocyte_positions, 'Greys')
        ax.set_title('Astrocytic Somata')

    def _laminar_density(self, points, ax):
        options = {'scale': 1, 'orientation': 'horizontal', 'n_bins': 20, 'measurement_function': 'mean'}
        from archngv.extras.plot.spatial_density_histogram import plot_laminar_mass_density
        plot_laminar_mass_density(ax, points, self.prop.bbox, options, {})

    def laminar_density_astrocytic_somata(self, ax):
        self._laminar_density(self.prop.astrocyte_positions, ax)

    def laminar_density_big_vessels(self, ax, cutoff_radius=6.):
        vasculature = self.prop.vasculature
        points = self.prop.vasculature.points
        radii = self.prop.vasculature.radii
        mask = radii > cutoff_radius
        self._laminar_density(points[mask], ax)

    def laminar_density_capillaries(self, ax, cutoff_radius=6.):
        vasculature = self.prop.vasculature
        points = self.prop.vasculature.points
        radii = self.prop.vasculature.radii
        mask = radii < cutoff_radius
        self._laminar_density(points[mask], ax)

    def spatial_density_neuronal_somata(self, ax):
        self._point_density_plot(ax, self.prop.neuronal_positions, 'Greys')

    def spatial_density_endfeet_targets(self, ax):
        self._point_density_plot(ax, self.prop.endfeet_surface_targets, 'Greys')
        ax.set_title('Endfeet Contacts')

    def spatial_density_big_vessels(self, ax, cutoff_radius=6.):
        vasculature = self.prop.vasculature
        points = self.prop.vasculature.points
        radii = self.prop.vasculature.radii
        mask = radii > cutoff_radius
        self._point_density_plot(ax, points[mask], 'Greys')
        ax.set_title('Large Vessels')

    def spatial_density_small_vessels(self, ax, cutoff_radius=6.):
        vasculature = self.prop.vasculature
        points = self.prop.vasculature.points
        radii = self.prop.vasculature.radii
        mask = radii < cutoff_radius
        self._point_density_plot(ax, points[mask], 'Greys')
        ax.set_title('Capillaries')

    def astrocyte_sizes_pial_layer_vs_deep(self):
        pass

    def neuron_stats_per_domain(self, index):

        for domain in self.prop.overlapping_microdomains:

            xmin, ymin, zmin, xmax, ymax, zmax = domain.bounding_box

            stats = index.intersection(xmin, ymin, zmin, xmax, ymax, zmax)
            
            print(stats)
