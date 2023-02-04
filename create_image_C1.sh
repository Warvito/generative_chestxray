#!/bin/bash
#
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
TAG=ldm_mimic_c1

docker build --network=host -f ./Dockerfile.c1 --tag "${TAG}" .
