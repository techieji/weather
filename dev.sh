#!/bin/bash
set -euxo pipefail

docker build --rm -f Dockerfile -t ml-nwp .
docker run --rm -it -v `pwd`:/root ml-nwp
