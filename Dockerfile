FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.7.0-cudnn8-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip git -y

# install torch
RUN pip3 install torch torchvision torchaudio
# install numerical
RUN pip3 install numpy scipy wheel matplotlib ipykernel ipywidgets autograd imageio
# install optimizer,scheduker
RUN pip3 install timm dadaptation
# install dependencies via pip
RUN pip3 install --upgrade pip

