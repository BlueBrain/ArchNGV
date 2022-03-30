Changelog
=========

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


.. _`NSETM-1830`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1830
.. _`NSETM-1778`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1778
.. _`NSETM-1807`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1807
.. _`NSETM-1746`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1746
.. _`NSETM-1768`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1768
.. _`NSETM-1740`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1740
