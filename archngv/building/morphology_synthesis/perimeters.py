""" Perimeter distribution on morphologies """
from collections import deque
import numpy as np
import morphio


class LinearRegressionNoiseModel:
    """
    Yi = slope * Xi + intercept + epsilon_i
    where epsilon_i is an iid following N(0.0, standard_deviation)

    Args:
        parameter_dict: dict
            Dictionary with the following key / value pairs:
                - slope: float
                - intercept : float
                - standard_deviation : float
        rng {None, int, RandomState, Generator}: Determines
            object used for drawing random numbers
    """
    def __init__(self, parameter_dict, rng):

        self.coefficients = np.array([float(parameter_dict['slope'])])
        self.intercept = float(parameter_dict['intercept'])

        self._loc = 0.0
        self._scale = parameter_dict['standard_deviation']
        self._rng = rng

    def _sample(self, size=None):
        """The gaussian noise generator"""
        return self._rng.normal(loc=self._loc, scale=self._scale, size=size)

    def _linear_function(self, values):
        """ Returns the linear prediction of the values """
        return np.dot(values, self.coefficients) + self.intercept

    def predict(self, input_values):
        """ Similar function signature with sklearn """

        n_values = len(input_values)
        noise_values = self._sample(size=n_values)
        predictions = self._linear_function(input_values)
        values = np.empty(n_values, dtype=np.float32)

        # ensure that we don't get negative values
        for i, (prediction, noise_value) in enumerate(zip(predictions, noise_values)):

            value = prediction + noise_value

            # resample until positive
            while value < 0.0:
                value = prediction + self._sample()

            values[i] = value

        return values


def _perimeters_upstream(section):
    """ Returns the perimeters upstream without taking into
    account the current section and accounting the duplicate
    points once.
    """
    iter_upstream = section.iter(morphio.IterType.upstream)
    next(iter_upstream)  # pylint: disable=stop-iteration-return

    for sec in iter_upstream:
        perimeters = sec.perimeters[:-1]
        for perimeter in perimeters[::-1]:
            yield perimeter


def _longest_downstream_leaf(section):
    """ Returns the leaf of the longest path downstream
    from the current section. The length of a section is determined
    by the size of its nodes (perimeters) """
    upstream_lengths = {section.id: 0}

    iter_sections = section.iter()
    next(iter_sections)

    longest_leaf = section
    longest_upstream_length = 0

    for sec in iter_sections:

        upstream_length = sec.perimeters.size + upstream_lengths[sec.parent.id]
        upstream_lengths[sec.id] = upstream_length

        if len(sec.children) == 0 and upstream_length > longest_upstream_length:
            longest_leaf = sec
            longest_upstream_length = upstream_length

    return longest_leaf


def _longest_downstream_path(section):
    """ The sections ids downstream without including
    the current section along the longest path"""
    leaf = _longest_downstream_leaf(section)

    downstream = deque([])
    for sec in leaf.iter(morphio.IterType.upstream):
        if sec.id == section.id:
            break
        downstream.appendleft(sec)

    return downstream


def _perimeters_downstream(section):
    """ Perimeters along the longest path downstream
    accounting the duplicate points once """
    for s in _longest_downstream_path(section):
        for perimeter in s.perimeters[1:]:
            yield perimeter


def _array_from_generator(array_size, value_generator):

    values = np.empty(array_size, dtype=np.float32)

    values[0] = next(value_generator)

    pos = 1
    for value in value_generator:
        if pos == array_size:
            break
        values[pos] = value
        pos += 1
    else:
        # if no other values exist, copy the last valid one
        values[pos:] = values[pos - 1]
    return values


def _reflect_perimeters(section, expansion_length):

    perimeters = np.empty(expansion_length, dtype=np.float32)

    section_perimeters = section.perimeters[1:]
    length = section_perimeters.size

    if expansion_length > section_perimeters.size:
        perimeters[:length] = section_perimeters
        perimeters[length:] = perimeters[length - 1]
    else:
        perimeters[:] = section_perimeters[:expansion_length]

    return perimeters


def _expand_start(section, expansion_length):
    """ Find the expansion for the section periemeters from the left by acessing sections
    upsream. If the root is reached then the first value is copied until the expansion_length
    is reached. If the section is root, the perimeters will be reflected at the first point of
    the array.
    """
    if section.is_root:
        return _reflect_perimeters(section, expansion_length)[::-1]
    return _array_from_generator(expansion_length, _perimeters_upstream(section))[::-1]


def _expand_end(section, expansion_length):
    """ Expands the end of the section perimeters by expansion_length. If the section has
    children then it expands towards the longest perimeters path downstream. If a termination
    is reached the last value will be copied until the expansion_length is reached.
    """
    if len(section.children) == 0:
        # terminations are extended to zero
        return np.full(expansion_length, fill_value=section.perimeters[-1], dtype=np.float32)

    return _array_from_generator(expansion_length, _perimeters_downstream(section))


def _predict_perimeters(section, statistical_model):
    """
    Args:
        morphology: morphio.mut.Section

        statistical_model:
            Model to predict the perimeters from the diameters. Required
            to have the predict method
    """
    diameters = section.diameters

    # we expand one dimension to be compatible with sklearn predict
    perimeters = statistical_model.predict(diameters[:, None])

    # non-roots have a duplicate point!
    if not section.is_root:
        perimeters[0] = section.parent.perimeters[-1]

    return perimeters


def _smooth_perimeters(section, smoothing_window):

    window_length = len(smoothing_window)

    # how much do we need to extend our array for a valid convolution
    # [1 [0, 1, 2, 3] 1] array with expansion for window of length 3
    # [1  1  1]          convolution window of length 3
    # Examples: window_length 3 -> expansion 1
    #           window_length 5 -> expansion 2
    expansion_length = window_length // 2

    # the start and end expansions
    end_expansion = _expand_end(section, expansion_length)
    beg_expansion = _expand_start(section, expansion_length)

    expanded_values = np.hstack((beg_expansion, section.perimeters, end_expansion))
    return np.convolve(smoothing_window, expanded_values, mode='valid')


def _smooth_morphology_perimeters(morphology, smoothing_window):

    # keep track of old and new total perimeters
    old_total = new_total = 0.0

    new_perimeters_list = []

    # updating should not be applied while smoothing the morphology
    # we have to smooth with respect to the current perimeters and not
    # update while we do that
    for section in morphology.iter():

        old_total += section.perimeters.sum()
        new_perimeters = _smooth_perimeters(section, smoothing_window)

        new_total += new_perimeters.sum()
        new_perimeters_list.append(new_perimeters)

    # we want to maintain the total perimeter after smoothing
    # therefore we will scale all new perimeters accordingly
    scaling_factor = old_total / new_total

    # with all new values available, it's time to update
    for section, new_perimeters in zip(morphology.iter(), new_perimeters_list):
        section.perimeters = new_perimeters * scaling_factor


def add_perimeters_to_morphology(morphology, parameters, rng):
    """
    Args:
        morphology: morphio.mut.Morphology
            Mutable morphology

        parameters: dict
            - statistical_model
                - slope
                - intercept
                - standard_deviation
            - smoothing
                - window

        rng {None, int, RandomState, Generator}: Determines
            object used for drawing random numbers
    """
    model = LinearRegressionNoiseModel(parameters['statistical_model'], rng)

    # first pass predict perimeters from diameters
    for section in morphology.iter():
        section.perimeters = _predict_perimeters(section, model)

    # normalized smoothing window from parameter json
    smoothing_window = np.asarray(parameters['smoothing']['window'], dtype=np.float32)
    smoothing_window /= smoothing_window.sum()

    # second pass smooth perimeters for continuity
    _smooth_morphology_perimeters(morphology, smoothing_window)
