#!/bin/sh

BIN=mfnwt
VERSION=1.1.4
ZIP=MODFLOW-NWT_${VERSION}.zip
ZIP_DIR=MODFLOW-NWT_${VERSION}
URL=https://water.usgs.gov/water-resources/software/MODFLOW-NWT/${ZIP}

. ci/do_install.sh
