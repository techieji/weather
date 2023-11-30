#!/bin/sh

docker build --rm -f Dockerfile -t ml-nwp .
docker run --rm -it -v `pwd`:/src ml-nwp