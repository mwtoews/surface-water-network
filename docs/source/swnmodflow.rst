SwnModflow
==========

.. currentmodule:: swn.modflow

Constructors
------------

.. autosummary::
   :toctree: ref/

   SwnModflow
   SwnModflow.from_swn_flopy
   SwnModflow.from_pickle

Properties
----------

.. autosummary::
   :toctree: ref/

   SwnModflow.model
   SwnModflow.segments
   SwnModflow.diversions
   SwnModflow.reaches
   SwnModflow.segment_data
   SwnModflow.segment_data_ts

Reach data generation
---------------------
.. autosummary::
   :toctree: ref/

   SwnModflow.set_reach_data_from_array
   SwnModflow.set_reach_data_from_segments
   SwnModflow.set_reach_slope

Segment data generation
-----------------------
.. autosummary::
   :toctree: ref/

   SwnModflow.new_segment_data
   SwnModflow.default_segment_data
   SwnModflow.set_segment_data_from_scalar
   SwnModflow.set_segment_data_from_segments
   SwnModflow.set_segment_data_from_diversions

Other packages
--------------
.. autosummary::
   :toctree: ref/

    SwnModflow.set_package_obj
    SwnModflow.package_period_frame
    SwnModflow.write_package_period
    SwnModflow.flopy_package_period

Utilities
---------
.. autosummary::
   :toctree: ref/

    SwnModflow.get_location_frame_reach_info
    SwnModflow.route_reaches
