#! /bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# apt-get update
# apt-get install -y python3 python3-pip cmake meson build-essential libeigen3-dev python3-dev

# build pybind11
# cd $SCRIPT_DIR
# cd third_party/pybind11
# rm -rf build
# mkdir build
# cd build
# cmake ..
# make -j

cd $SCRIPT_DIR
rm -rf build
meson build
cd build
ninja
cp ./libdp.so ../dprofile
