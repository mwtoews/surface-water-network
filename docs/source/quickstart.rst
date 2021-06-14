Quickstart guide
################

The aim of this guide is to provide a quick overview of the capabilities of this package to read GIS inputs to create a surface water network.


Surface water network
*********************

This example is based on `this image
<https://commons.wikimedia.org/wiki/File:Flussordnung_(Strahler).svg>`_ that
illustrates Strahler number.

.. ipython:: python

    import swn
    import geopandas
    from shapely import wkt

    lines = geopandas.GeoSeries(list(wkt.loads("""\
    MULTILINESTRING(
        (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
        (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
        (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
        (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
        (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
        (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
        (770 100, 820  40))""").geoms))
    lines.index += 100
    lines.index.name = "idx"

Create a surface water network object from the lines using
:py:func:`SurfaceWaterNetwork.from_lines`, then illustrate a few properties:

.. ipython:: python

    n = swn.SurfaceWaterNetwork.from_lines(lines)
    print(n)
    print(n.segments[["to_segnum", "from_segnums", "stream_order", "num_to_outlet"]])
    @savefig fluss_swn.png
    n.plot();


