from .common import subsample, remove_all_spines

import seaborn as sns
import pylab as plt

def vasculature_contacts_comparison_figure(vasculature, graph_targeting_results, bounding_box, max_samples=20000):


    def _density_plot(ax, points, Nsamples, cmap):

        sample = points if points.shape[0] < Nsamples else subsample(points, Nsamples)

        sns.kdeplot(sample[:, 0], sample[:, 1], ax=ax, shade=True, cmap=cmap)

        ax.set_xlabel('x (um)')


    x_range, y_range, z_range = bounding_box.ranges.T

    f, axes = plt.subplots(1, 3, figsize=(15, 10))

    contact_points = graph_targeting_results['points']

    ax1 = axes[0]
    _density_plot(ax1, vasculature.points[vasculature.radii <= 6.], max_samples, 'Blues')
    ax1.set_title('Capillaries')

    ax2 = axes[1]
    _density_plot(ax2, vasculature.points[vasculature.radii > 6.], max_samples, 'Reds')
    ax2.set_title('Large Vessels')


    ax3 = axes[2]
    _density_plot(ax3, contact_points, max_samples, 'Greys')
    ax3.set_title('Endfeet')

    for axis in [ax1, ax2, ax3]:

        axis.set_xlim(x_range)
        axis.set_ylim(y_range)
        axis.invert_yaxis()
        remove_all_spines(axis)

    return f

"""
def endfeet_statics_figure():

    f, ax = plt.subplots(2, 4, figsize=(20, 20), gridspec_kw={'height_ratios':[4, 1]})



    ax0 = ax[0, 0]
    vasc_bin_centers = bin_centers()
ax0.plot(1e6 * vasc_dens, vasc_bin_centers, linewidth=1., color='r')
ax0.plot(1e6 * endf_dens, vasc_bin_centers, linewidth=1., color='k')

ax0.set_xlabel('mm / (mm^3)')
ax0.set_ylabel('Cortical Depth (um)')

xmin, xmax = ax0.get_xlim()


ax0.set_title('Length Density')
ax0.hlines(layers, xmin, xmax, alpha=0.2, linestyle='--')

# -----------------------------------------------------------------------------------------------

ax1 = ax[0, 1]

endf_con_bin_centers = endfeet_contact_bins[:-1] + (endfeet_contact_bins[1:] - endfeet_contact_bins[:-1]) * 0.5

ax1.plot(1e9 * endfeet_contact_density, endf_con_bin_centers, linewidth=1., color='k')

ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xlabel('endfeet / (mm^3)')

xmin, xmax = ax1.get_xlim()
ax1.hlines(layers, xmin, xmax, alpha=0.2, linestyle='--')

ax1.set_ylim(ax1.get_ylim())
ax1.set_title('Endfeet Density')

# -----------------------------------------------------------------------------------------------

ax2 = ax[0, 2]

area_bin_centers = area_bins[:-1] + 0.5 * (area_bins[1:] - area_bins[0:-1])
ax2.plot(area_means, area_bin_centers, color='k', linewidth=1.)
#ax[2].fill_between(area_means - area_stds, area_means + area_stds, alpha=0.6, color='gray')
ax2.set_title('Endeet area distribution')

ax2.set_xlabel('Mean area (um^2)')

xmin, xmax = ax2.get_xlim()
ax2.hlines(layers, xmin, xmax, alpha=0.2, linestyle='--')

# -----------------------------------------------------------------------------------------------

ax3 = ax[0, 3]

sep_bin_centers = separation_bins[:-1] + 0.5 * (separation_bins[1:] - separation_bins[:-1])
ax3.plot(separation_means, sep_bin_centers, color='b', linewidth=1.)

ax3.set_title('Astrocyte - Capillary\nSeparation')
ax3.set_xlabel('(um)')

xmin, xmax = ax3.get_xlim()
ax3.hlines(layers, xmin, xmax, alpha=0.2, linestyle='--')

'''
for axis in [ax1, ax2, ax3]:
    axis.yaxis.set_visible(False)
'''
# -----------------------------------------------------------------------------------------------
ax4 = ax[1, 0]

ax4.hist(1e6 * endf_dens, histtype='step', color='k', bins=10, normed=True)
ax4.hist(1e6 * vasc_dens, histtype='step', color='r', bins=10, normed=True)

ax5 = ax[1, 1]
ax5.hist(1e6 * endfeet_contact_density, histtype='step', color='k', bins=10, normed=True)

ax6 = ax[1, 2]
ax6.hist(areas, histtype='step', color='k', bins=20, normed=True)

ax7 = ax[1, 3]
ax7.hist(separations, histtype='step', color='b', bins=20, normed=True, alpha=1)

for axis in [ax0, ax1, ax2, ax3]:
    axis.set_ylim(y_range)
    axis.invert_yaxis()

plt.legend()
plt.tight_layout()
"""
