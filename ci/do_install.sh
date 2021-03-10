#!/bin/sh
set -e

# This script is called from another install.sh script
if [ -z ${URL+x} ] || [ -z ${ZIP+x} ] || [ -z ${ZIP_DIR+x} ] || [ -z ${BIN+x} ]; then
   echo "This script cannot be called directly" 1>&2
   exit 1
elif [ -z "${INSTALL_PREFIX}" ]; then
    echo "INSTALL_PREFIX must be set"
    exit 1
fi

NPROC=2
UNAME="$(uname)" || UNAME=""
case ${UNAME} in
    Linux)
        NPROC=$(nproc) ;;
    Darwin)
        NPROC=$(sysctl -n hw.ncpu) ;;
esac
export MAKEFLAGS="-j ${NPROC}"

REPO_DIR=$(pwd)
INSTALLED_BIN=${INSTALL_PREFIX}/bin/${BIN}

if [ -d "${INSTALLED_BIN}" ]; then
    echo "Using cached install ${INSTALLED_BIN}"
else
    echo "Building ${INSTALLED_BIN}"
    echo "Downloading ${URL}"
    wget -nv ${URL} -O ${ZIP}
    echo "Extracting ${ZIP}"
    unzip -q ${ZIP}
    cd ${ZIP_DIR}
    cp ${REPO_DIR}/ci/${BIN}/CMakeLists.txt .
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -D CMAKE_MODULE_PATH=${REPO_DIR}/ci/cmake \
          ..
    make
    make install
fi
