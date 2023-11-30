#!/bin/bash
set -euxo pipefail

# This script sets up a minimal environment (untested) for ML-NWP models

sudo apt-get update && apt-get install -y gfortran git vim python3 python3-pip
python3 -m pip install -r requirements.txt
# sudo apt-get install -y libproj-dev proj-data proj-bin
sudo apt-get install -y libgeos-dev libnetcdf-dev libnetcdff-dev

if [ ! -d "pySPEEDY" ]; then
    git clone https://github.com/aperezhortal/pySPEEDY.git
fi
export NETCDF=/usr
cd pySPEEDY
#make
./build.sh
make test