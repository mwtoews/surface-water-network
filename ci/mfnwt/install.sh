#!/bin/bash
set -e

BIN=mfnwt
VER=1.1.4
ZIP=MODFLOW-NWT_$VER.zip
ZIPDIR=MODFLOW-NWT_$VER
URL=https://water.usgs.gov/water-resources/software/MODFLOW-NWT/$ZIP

source ci/do_install.sh
