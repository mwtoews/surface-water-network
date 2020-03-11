#!/bin/bash
set -e

# This script is called from another install.sh script
if [ -z ${URL+x} ]; then
   echo "This script cannot be called directly" 1>&2
   exit 1
fi

REPODIR=$(pwd)
PREFIX=$HOME/.local

if [ -d "$PREFIX/bin/$BIN" ]; then
    echo "Using cached install $PREFIX/bin/$BIN"
else
    echo "Building $PREFIX/bin/$BIN"
    wget -nv --show-progress $URL -O $ZIP
    unzip -q $ZIP
    cd $ZIPDIR
    cp $REPODIR/ci/$BIN/CMakeLists.txt .
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MODULE_PATH=$REPODIR/ci/cmake ..
    make -j2
    make install
fi
