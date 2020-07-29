"""Morphology synthesis data structures"""
from collections import namedtuple


EndfeetData = namedtuple('EndfeetData', ['targets', 'area_meshes'])

AstrocyteProperties = namedtuple('AstrocyteProperties', ['name', 'soma_position', 'soma_radius', 'microdomain'])

SpaceColonizationData = namedtuple(
    'SpaceColonizationData',
    ['point_cloud', 'influence_distance_factor', 'kill_distance_factor'])

EndfeetAttractionData = namedtuple(
    'EndfeetAttractionData',
    ['targets', 'field_function'])

TNSData = namedtuple('TNSData', ['parameters', 'distributions', 'context'])

SynthesisInputPaths = namedtuple('SynthesisInputPaths', [
    'cell_data',
    'microdomains',
    'synaptic_data',
    'gliovascular_data',
    'gliovascular_connectivity',
    'neuroglial_connectivity',
    'tns_parameters',
    'tns_distributions',
    'tns_context',
    'morphology_directory',
    'endfeet_areas'])