import numpy as np
from scipy.interpolate import interp1d

LAYERS = {'bins': np.array([0.0,
                            674.68206269999996,
                            1180.8844627000001,
                            1363.6375343,
                            1703.8656135000001,
                            1847.3347831999999,
                            2006.3482524000001]),
          'labels': ('VI', 'V', 'IV', 'III', 'II', 'I'),
          'centers': np.array([337.34103135,
                               927.7832627,
                               1272.2609985,
                               1533.7515739,
                               1775.60019835,
                               1926.8415178])
}

def plot_oriented(axis, x, y, orientation, **kwarg):

    if orientation == 'horizontal':

        axis.plot(x, y, **kwarg)

    elif orientation == 'vertical':

        axis.plot(y, x, **kwarg)

    else:

        msg = 'Unknown orientation {}'.format(orientation)
        raise BaseException(msg)


def smooth_convolve(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def smooth(x, y, n_points):

    ynew = np.linspace(y.min(), y.max(), n_points)

    xnew = interp1d(y, x)(ynew)

    return xnew, ynew


def bin_centers(bins):
    return bins[:-1] + 0.5 * (bins[1:] - bins[:-1])


def subsample(dataset, Nsamples):

    n_samples = dataset.shape[0]

    idx = np.arange(n_samples, dtype=np.uintp)

    sub_idx = np.random.choice(idx, Nsamples, replace=False)

    return dataset[sub_idx]

def remove_spines(axis, orientation_flags):
    " (right, bottom, left, top)"

    orientation_str = ['right', 'bottom', 'left', 'top']

    for ori, flag in zip(orientation_str, orientation_flags):
        axis.spines[ori].set_visible(flag)


def remove_top_right_spines(axis):

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)


def remove_all_spines(axis):

    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['top'].set_visible(False)


def add_layers(axis, layers, orientation='vertical', reflect=False, plot_options={}):

    bin_layers = layers['bins']
    str_layers = layers['labels']

    cnt_layers = layers['centers']

    if orientation is 'horizontal':

        (xmin, xmax) = axis.get_xlim()

        axis.hlines(bin_layers, xmin, xmax, **plot_options)
        axis.set_yticks(cnt_layers)
        axis.set_yticklabels(str_layers)

        if reflect:
            axis.set_label_position("right")

    elif orientation is 'vertical':

        (ymin, ymax) = axis.get_ylim()

        axis.vlines(bin_layers, ymin, ymax, **plot_options)
        axis.set_xticks(cnt_layers)
        axis.set_xticklabels(str_layers)

        if reflect:
            axis.set_label_position('up')

    else:

        raise TypeError('Unknown Orientation. Try horizontal or vertical')


import matplotlib.pyplot as plt
import matplotlib.offsetbox
from matplotlib.lines import Line2D
import numpy as np

class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
                 frameon=True, **kwargs):

        if ax is None:
            ax = plt.gca()

        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **kwargs)
        vline1 = Line2D([0,0],[0.,extent/2.], **kwargs)
        vline2 = Line2D([size,size],[0.,extent/2.], **kwargs)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
                                 align="center", pad=ppad, sep=sep) 
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon)

