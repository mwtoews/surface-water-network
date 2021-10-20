Change log
==========

Version 0.4
-----------

:Date: 20 October 2021

Breaking changes
~~~~~~~~~~~~~~~~
- Change ``SurfaceWaterNetwork(lines.geometry)`` -> ``SurfaceWaterNetwork.from_lines(lines.geometry)``
- Change ``MfSfrNetwork(n, m, ...)`` -> ``SwnModflow.from_swn_flopy(n, m)``
- Legacy ``swn.modflow.MfSfrNetwork`` kept, but will be dropped next release

New things
~~~~~~~~~~
- Add ``.to_pickle()`` and ``.from_pickle()`` methods
- Support MODFLOW 6 models (as well as classic MODFLOW)
- Add ``segments_series`` and ``pair_segments_frame`` methods to ``SurfaceWaterNetwork``
- Add ``swn.compat`` module for compatibility
- Add Sphinx documentation, with doctests

Other changes
~~~~~~~~~~~~~
- Improve performance of ``SurfaceWaterNetwork.from_lines``
- Add ``mult`` multiplier to ``topnet2ts``
- Use declarative configuration for project setup, remove ``setup.py``

Version 0.3
-----------

:Date: 10 March 2021

- Add GitHub Actions for testing
- Remove GDAL dependency
- Require pyproj>=2.0
- Add several methods in ``swn.spatial``: ``compare_crs``, ``get_crs``, ``find_segnum_in_swn``

Version 0.2
-----------

:Date: 11 December 2019

- Create package setup, minimum version published on PyPI
- Add citation based on New Zealand Hydrological Society Conference
- Reorganised into modular structure
- Added ``plot()`` functions
- Added string representations of main objects

Version 0.1
-----------

:Date: 6 November 2019

- Initial release
- No license or package setup
