Changelog
=========

Version 2.0.2
-------------

Improvements
~~~~~~~~~~~~

- Simplified NGV exporters and removed a lot of redundant code. [`NSETM-1927`_]
- Make the NGV population types consistent with the SONATA documentation, and maintain backward
  backward compatibility with the legacy types. [`NSETM-1923`_]

Fixed
~~~~~

- Output the correct key entry 'endfeet_meshes_file' for the endfeet meshes path in the SONATA
  config. Maintain backward compatibility with the old 'endfeet_meshes' entry. [`NSETM-1904`]


Version 2.0.1
-------------

New Features
~~~~~~~~~~~~

- NGV building supports circuit-build manifests, where the ngv specific configuration has to be
  under a root entry 'ngv', [`NSETM-1872`_]

Version 2.0.0
-------------

Improvements
~~~~~~~~~~~~

- Endfeet meshes file layout has been converted to the grouped properties one. [`NSETM-1830`_]
- Merge microdomain files into a single file at the build root level. [`NSETM-1807`_]
- Use a generic file layout for grouped properties to store the microdomains that can be extended
  with and arbitrary number of additional properties. [`NSETM-1807`_]
- Multiple yaml configuration files merged into the main MANIFEST.yaml. [`NSETM-1746`_]
- Refactored the neuroglial connectivity out of ngv.py to reduce its size. [`NSETM-1768`_]
- Refactored the gliovascular connectivity out of ngv.py to reduce its size.
- The app cli has been compressed into the ngv.py module. [`NSETM-1740`_]

New Features
~~~~~~~~~~~~
- Resampling was added in synthesis to create an upper bound in morphology points [`NSETM-1778`_]. 

Removed
~~~~~~~

- SONATA config no longer stores the bioname parameters. [`NSETM-1746`_]
- The deprecated `extras` subpackage has been removed.

.. _`NSETM-1927`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1927
.. _`NSETM-1923`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1923
.. _`NSETM-1904`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1904
.. _`NSETM-1872`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1872
.. _`NSETM-1830`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1830
.. _`NSETM-1778`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1778
.. _`NSETM-1807`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1807
.. _`NSETM-1746`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1746
.. _`NSETM-1768`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1768
.. _`NSETM-1740`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1740
