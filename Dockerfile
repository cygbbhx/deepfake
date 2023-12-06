FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* 


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /workspace
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

# Install Python packages
RUN pip3 install numpy matplotlib librosa h5py pandas Pillow omegaconf 

RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python

RUN pip3 install facenet-pytorch
RUN pip3 install jsbeautifier

# Copy the files from the current directory into the image
COPY . .
