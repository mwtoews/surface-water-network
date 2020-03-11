#!/bin/bash
set -e

BIN=mf2005
VER=1_12u
ZIP=MF2005.$VER.zip
ZIPDIR=MF2005.$VER
URL=https://water.usgs.gov/water-resources/software/MODFLOW-2005/$ZIP

source ci/do_install.sh
