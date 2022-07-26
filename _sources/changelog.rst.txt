Change log
==========

.. currentmodule:: swn

Version 0.5
-----------

:Date: 20 July 2022

Breaking changes
~~~~~~~~~~~~~~~~
- Minimum requirements are Python 3.7, pandas 1.2 (:pr:`38`)
- Add outside segnum to modflow if it is downstream from others (:pr:`50`)
- Change behaviour of :py:meth:`SurfaceWaterNetwork.set_diversions` which by default will now match to the closest segment line (:pr:`52`)

New things
~~~~~~~~~~
- Add :py:meth:`SurfaceWaterNetwork.locate_geoms` method (:pr:`41`, :pr:`48`, :pr:`51`)
- Add ``run`` option to :py:func:`~swn.file.topnet2ts`, improve error messages (:pr:`44`)
- Add routing methods: :py:meth:`SurfaceWaterNetwork.route_segnums`, :py:meth:`modflow.SwnModflow.route_reaches`, and :py:meth:`modflow.SwnMf6.route_reaches` (:pr:`45`)
- Add :py:mod:`~swn.spatial` module functions: :py:func:`~swn.spatial.find_location_pairs` and :py:func:`~swn.spatial.location_pair_geoms` (:pr:`46`)
- Add methods :py:meth:`modflow.SwnModflow.get_location_frame_reach_info` and :py:meth:`modflow.SwnMf6.get_location_frame_reach_info` (:pr:`49`)
- Implement diversions for MODFLOW6 (:pr:`54`)

Other changes
~~~~~~~~~~~~~
- Add packing dependency to check package version (:pr:`37`)
- Convert class attributes to properties for :py:attr:`modflow.SwnModflow.segment_data` and :py:attr:`~swn.modflow.SwnModflow.segment_data_ts` (:pr:`42`)
- Convert class attributes to properties for :py:attr:`modflow.SwnMf6.segments`, :py:attr:`~swn.modflow.SwnMf6.diversions`, and :py:attr:`~swn.modflow.SwnMf6.reaches` (:pr:`43`)
- Rename ``SurfaceWaterNetwork.query()`` → :py:meth:`~swn.SurfaceWaterNetwork.gather_segnums`; show ``DeprecationWarning`` with former method (:pr:`47`)
- Move project metadata to ``pyproject.toml``, add a few optional dependencies (:pr:`53`)

Version 0.4
-----------

:Date: 20 October 2021

Breaking changes
~~~~~~~~~~~~~~~~
- Change ``SurfaceWaterNetwork(lines.geometry)`` → :py:meth:`SurfaceWaterNetwork.from_lines(lines.geometry) <swn.SurfaceWaterNetwork.from_lines>`
- Change ``MfSfrNetwork(n, m, ...)`` → :py:meth:`SwnModflow.from_swn_flopy(n, m) <swn.modflow.SwnModflow.from_swn_flopy>`
- Legacy ``modflow.MfSfrNetwork`` kept, but will be dropped by version 0.6

New things
~~~~~~~~~~
- Add ``.to_pickle()`` and ``.from_pickle()`` methods to core classes
- Support MODFLOW 6 models with :py:class:`modflow.SwnMf6` (:pr:`12`)
- Add :py:meth:`SurfaceWaterNetwork.segments_series` and :py:meth:`~swn.SurfaceWaterNetwork.pair_segments_frame` methods (:pr:`15`)
- Add methods for setting stream bed elevations (:pr:`27`)
- Add :py:mod:`~swn.compat` module for compatibility
- Add Sphinx documentation, with doctests (:pr:`18`)

Other changes
~~~~~~~~~~~~~
- Improve performance of :py:meth:`SurfaceWaterNetwork.from_lines` (:pr:`33`)
- Add ``mult`` multiplier to :py:func:`~swn.file.topnet2ts`
- Use declarative configuration for project setup, remove ``setup.py`` (:pr:`35`)

Version 0.3
-----------

:Date: 10 March 2021

- Add GitHub Actions for testing (:pr:`9`)
- Remove GDAL dependency
- Require pyproj>=2.0
- Add several functions to :py:mod:`~swn.spatial`: :py:func:`~swn.spatial.compare_crs`, :py:func:`~swn.spatial.get_crs`, :py:func:`~swn.spatial.find_segnum_in_swn`

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
