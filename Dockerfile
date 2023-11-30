FROM python:3.11
LABEL maintainer="Pradhyum R"
LABEL description="Sets up a minimal development environment for ML-NWP models"
RUN apt-get update && apt-get install -y gfortran git vim
RUN apt-get install -y libgeos-dev libnetcdf-dev libnetcdff-dev
ENV NETCDF=/usr
RUN git clone https://github.com/aperezhortal/pySPEEDY.git
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN cd pySPEEDY; ./build.sh
CMD ["bash"]