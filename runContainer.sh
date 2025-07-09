#!/bin/sh

# docker pull quay.io/jupyter/scipy-notebook:2024-10-03

# mkdir -p work

docker run -ti \
    --name project \
    --memory 10g  \
    -v .:/home/jovyan/work/ \
    -e JUPYTER_TOKEN="<token>" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -p 8889:8888 \
    quay.io/jupyter/scipy-notebook:2024-10-03 
