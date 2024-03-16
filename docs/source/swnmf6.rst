SwnMf6
======

.. currentmodule:: swn.modflow

Constructors
------------

.. autosummary::
   :toctree: ref/

   SwnMf6
   SwnMf6.from_swn_flopy
   SwnMf6.from_pickle

Properties
----------

.. autosummary::
   :toctree: ref/

   SwnMf6.model
   SwnMf6.segments
   SwnMf6.diversions
   SwnMf6.reaches

Reach data generation
---------------------
.. autosummary::
   :toctree: ref/

   SwnMf6.set_reach_data_from_array
   SwnMf6.set_reach_data_from_segments
   SwnMf6.set_reach_slope

Streamflow Routing Package
--------------------------
.. autosummary::
   :toctree: ref/

   SwnMf6.default_packagedata
   SwnMf6.packagedata_frame
   SwnMf6.write_packagedata
   SwnMf6.flopy_packagedata
   SwnMf6.connectiondata_series
   SwnMf6.write_connectiondata
   SwnMf6.flopy_connectiondata
   SwnMf6.diversions_frame
   SwnMf6.write_diversions
   SwnMf6.flopy_diversions
   SwnMf6.set_sfr_obj

Other packages
--------------
.. autosummary::
   :toctree: ref/

   SwnMf6.set_package_obj
   SwnMf6.package_period_frame
   SwnMf6.write_package_period
   SwnMf6.flopy_package_period

Utilities
---------
.. autosummary::
   :toctree: ref/

    SwnMf6.get_location_frame_reach_info
    SwnMf6.gather_reaches
    SwnMf6.route_reaches
