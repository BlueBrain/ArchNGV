Bioname: ngv circuit build recipes
==================================


Bioname
-------

`bioname` is the folder where all the circuit building datasets are stored.

An ngv circuit `bioname` includes the following files:

* ``MANIFEST.yaml``
* ``cell_placemement.yaml``
* ``gliovascular_connectivity.yaml``
* ``endfeet_area.yaml``
* ``microdomains.yaml``
* ``synthesis.yaml``
* ``tns_context.json``
* ``tns_distributions.json``
* ``tns_parameters.json``
* ``astrocyte_gap_junction_recipe.xml``


MANIFEST.yaml
~~~~~~~~~~~~~

This is the main configuration for the circuit build defining:

* atlas space
* vasculature morphology and mesh
* base neuronal circuit
* individual task parameters

It starts with a `common` section:

.. code-block:: yaml

    common:
      parallel: true
      log_level: 'WARNING'
      seed: 0
      atlas: '/gpfs/bbp.cscs.ch/project/proj62/NGV/atlases/O1-230/20190517'
      vasculature: '/gpfs/bbp.cscs.ch/project/proj62/NGV/ARCHNGV/Data/vasculature_datasets/raw_pruned_cap_circuit_coo_inscribed_spec_atlas_translated_new_spec.h5'
      vasculature_mesh: '/gpfs/bbp.cscs.ch/project/proj62/NGV/ARCHNGV/Data/vasculature_datasets/raw_pruned_cap_circuit_coo_inscribed_spec_atlas_translated_triangulated_zup_fixed.obj'
      base_circuit: '/gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/CircuitConfig'
      base_circuit_sonata: '/gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/circuit_config.json'
      base_circuit_cells: '/gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/networks/nodes/All/nodes.h5'
      base_circuit_connectome: '/gpfs/bbp.cscs.ch/project/proj62/Circuits/O1/20190912_spines/sonata/networks/edges/functional/All/edges.h5'

and follows with separate sections for each phase.

We'll provide a short description for eac of the `common` values here.

**parallel**
    Flag for enabling parallel running.

**log_level**
    Logging level to be used for the build.

**seed**
    Random seed to be used for the build.

**atlas**
    Folder path of atlas

**vasculature**
    Path to vasculature morphology.

**vasculature_mesh**
    Path to vasculature watertight surface mesh.

**base_circuit**
    Path to the CircuitConfig file of the neuronal circuit that will be used for the ngv building.

**base_circuit_sonata**
    Path to the sonata config of the neuronal circuit that will be used for the ngv building.

**base_circuit_cells**
    Path to the sonata node population file of neurons.

**base_circuit_connectome**
    Path to the sonata edge population file of the synaptic connectivity.

assign_emodels
~~~~~~~~~~~~~~

Assign emodels parameters determine the electrical models that are going to be used in the building.

.. code-block:: yaml

    assign_emodels:
      templates_dir: '/gpfs/bbp.cscs.ch/project/proj62/NGV/emodels'
      hoc_template: 'astrocyte'

**templated_dir**
    Directory for the electrical models

**hoc_template**
    Template that will be used for astrocytes

cell_placement
~~~~~~~~~~~~~~~~~~~

Cell placement parameters determine the spatial density and disperion of astrocytic somata.

.. code-block:: yaml

    density: '[density]astrocytes'

    soma_radius: [5.6, 0.74, 0.1, 20]

    Energy:
        potentials:
            spring: [32.0, 1.0]

    MetropolisHastings:
        n_initial: 10
        beta: 0.01
        ntrials: 3
        cutoff_radius: 60.0


**density**
    The nrrd density dataset to choose from the atlas folder.

**soma_radius**
    Normal distribution of soma size (mean, std, min, max)

**Energy**
    The repulsion potential to use for evenly spacing astrocytic somata

**MetropolisHastings**
    Parameters for the Metropolis-Hastings algorithm.

microdomains
~~~~~~~~~~~~

Microdomains are the bounding polygons that determine the extent of astrocytes.

.. code-block:: yaml

    overlap_distribution:
      type: normal
      values: [0.1, 0.001]

**overlap_distribution**
    Distribution of the overlapping volume of the microdomains.


synthesis
~~~~~~~~~

Morphology synthesis of astrocytes. `tns_parameters`, `tns_distributions`, `tns_context` will be explained below. Here only the perimeter statistic model is determined.

.. code-block:: yaml

    perimeter_distribution:
        enabled: true
        statistical_model:
            slope: 2.060005867796768
            intercept: 1.0219733661696733
            standard_deviation: 1.1161359624857308
        smoothing:
            window: [1.0, 1.0, 1.0, 1.0, 1.0]

**perimeter_distribution**
    The statistical model to predict the perimeters on the synthesized morphologies based on their diameters.

gliovascular_connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Gliovascular connectivity establises connections between astrocytes and the vasculature based on the microdomain of each astrocyte. Morphology has not been generated in this step.

.. code-block:: yaml

    graph_targeting:
        linear_density: 0.17

    connection:
        reachout_strategy: 'maximum_reachout'
        endfeet_distribution: [2, 2, 0, 15]


**graph_targeting***
    During graph targeting points are evenly distributed on the skeleton of the vasculature, with a linear density specified above.

**connection**
    To establish connection between astrocytes and the vasculature, the reachout strategy needs to be specified as well as the distribution
    of endfeet per astrocyte.

endfeet_area
~~~~~~~~~~~~

Parameters for the surface growing of astrocytic endfeet.

.. code-block:: yaml

    fmm_cutoff_radius: 1000.
    area_distribution: [200.0, 10.0, 0.0, 1000.0]
    thickness_distribution: [1.0, 0.1, 0.01, 2.0]

**fmm_cutoff_radius**
    The maximum radius that the growing of an endfoot with stop.

**area_distribution**
    Distribution of the endfeet areas for pruning (mean, sdev, min, max).

**thickness_distribution**
    Distribution of the thickness of each astrocyte endfoot (mean, sdev, min, max).

tns_context.json
~~~~~~~~~~~~~~~~

The context specifies the spatial parameters that will be used in morphology synthesis.

.. code-block:: json

    {
      "field": {
        "type": "logit",
        "slope": 0.11832134,
        "intercept": 0.36720545
      },
      "space_colonization": {
        "kill_distance_factor": 15.0,
        "influence_distance_factor": 45.0
      }
    }

The space colonization algorithm uses an attraction field for each endfoot target, so that processes grow towards the vascular sites. Furthermore, the `kill_distance_factor` and
`influence_distance_factor` after multiplied with the segment length, determine the distance for removal and attraction of resources respectively.

tns_distributions.json
~~~~~~~~~~~~~~~~~~~~~~

Distributions include the barcode sampling and statistical distributions required for morphology synthesis.

tns_parameters.json
~~~~~~~~~~~~~~~~~~~

The parameters for morphology synthesis.

.. code-block:: json

    {
      "basal": {
        "randomness": 0.3,
        "targeting": 0.01,
        "radius": 0.3,
        "orientation": null,
        "growth_method": "tmd_space_colonization",
        "branching_method": "bio_oriented",
        "modify": null,
        "step_size": {
          "norm": {
            "mean": 0.1,
            "std": 0.001
          }
        },
        "metric": "path_distances",
        "tree_type": 3,
        "filtration_metric": "path_distance",
        "barcode_scaling": false
      },
      "apical": {},
      "axon": {
        "randomness": 0.25,
        "targeting": 0.07,
        "radius": 0.3,
        "orientation": null,
        "growth_method": "tmd_space_colonization_target",
        "branching_method": "bio_oriented",
        "modify": null,
        "step_size": {
          "norm": {
            "mean": 0.1,
            "std": 0.001
          }
        },
        "metric": "path_distances",
        "tree_type": 2,
        "filtration_metric": "path_distance",
        "bias": 0.9,
        "barcode_scaling": false
      },
      "origin": [
        0.0,
        0.0,
        0.0
      ],
      "grow_types": [
        "basal",
        "axon"
      ],
      "diameter_params": {
        "mtypes_file": null,
        "models": [
          "astrocyte"
        ],
        "neurite_types": [
          "basal",
          "axon"
        ],
        "terminal_threshold": 2.0,
        "taper": {
          "max": 1.1,
          "min": -1.1
        },
        "asymmetry_threshold": {
          "axon": 0.05,
          "basal": 0.05
        },
        "n_samples": 1,
        "seed": 0,
        "trunk_max_tries": 100,
        "n_cpu": 1,
        "method": "external"
      }
    }


astrocyte_gap_junction_recipe.xml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the touchdetector's recipe for establishing gap junctional touches between astrocytes and neighboring astrocytes.
