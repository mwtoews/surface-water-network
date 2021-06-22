SurfaceWaterNetwork
===================

.. currentmodule:: swn

Constructors
------------

.. autosummary::
   :toctree: ref/

   SurfaceWaterNetwork
   SurfaceWaterNetwork.from_lines
   SurfaceWaterNetwork.from_pickle

DataFrame properties
--------------------

.. autosummary::
   :toctree: ref/

   SurfaceWaterNetwork.segments
   SurfaceWaterNetwork.catchments
   SurfaceWaterNetwork.diversions

Other properties
----------------

.. autosummary::
   :toctree: ref/

   SurfaceWaterNetwork.has_z
   SurfaceWaterNetwork.headwater
   SurfaceWaterNetwork.outlets
   SurfaceWaterNetwork.to_segnums
   SurfaceWaterNetwork.from_segnums

Methods
-------

.. autosummary::
   :toctree: ref/

   SurfaceWaterNetwork.segments_series
   SurfaceWaterNetwork.pair_segments_frame
   SurfaceWaterNetwork.accumulate_values
   SurfaceWaterNetwork.query
   SurfaceWaterNetwork.aggregate
   SurfaceWaterNetwork.evaluate_upstream_length
   SurfaceWaterNetwork.evaluate_upstream_area
   SurfaceWaterNetwork.estimate_width
   SurfaceWaterNetwork.adjust_elevation_profile

Miscellaneous
-------------

.. autosummary::
   :toctree: ref/

   SurfaceWaterNetwork.plot
   SurfaceWaterNetwork.to_pickle

