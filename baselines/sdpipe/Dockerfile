# base docker image

# Known issues:
# - hetu segfault: compile with proper cmake, make sure protoc/libprotobuf matches
# - glibc version mismsatch: use old ubuntu
# - NCCL unhandled system error: pass --shm-size to `docker run`

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04   # glibc version mismatch
# FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu20.04   # NCCL unhandled system error

# init workdir
RUN mkdir -p /build
WORKDIR /build

# install common tool & conda
RUN apt update && \
    apt install wget git vim -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# install conda alpa env
# Note: must use hetu env cmake, otherwise protobuf version mismatch causes segfault
RUN . /opt/conda/etc/profile.d/conda.sh && \
    git clone --recursive https://github.com/Hsword/VLDB2023_SDPipe && \
    cd VLDB2023_SDPipe && \
    conda env create -f environment.yml && \
    conda activate hetu && \
    conda install -y cmake boto3 requests && \
    pip3 install torch torchvision more-itertools h5py tensorboard && \
    cp cmake/config.example.cmake cmake/config.cmake && \
    mkdir build && cd build && /opt/conda/envs/hetu/bin/cmake .. && make -j && cd .. && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate hetu" >> ~/.bashrc && \
    echo ". /build/VLDB2023_SDPipe/hetu.exp" >> ~/.bashrc
