FROM python:3.11
LABEL maintainer="Pradhyum R"
LABEL description="Sets up a minimal development environment for ML-NWP models"
EXPOSE 8888
RUN apt-get update && apt-get install -y gfortran git vim curl
RUN apt-get install -y libgeos-dev libnetcdf-dev libnetcdff-dev
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# To download the boundary conditions (TODO: get a better method)
RUN git clone https://github.com/aperezhortal/pySPEEDY.git

WORKDIR /root
# CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
CMD bash
