#!/bin/bash
set -e

IMAGE="pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel"
CONTAINER_NAME="fcmae"

# 1. Pull the base image
echo "[1/4] Pulling Docker image..."
docker pull $IMAGE

# 2. Create the container (only if it doesn't exist yet)
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "[2/4] Container '${CONTAINER_NAME}' already exists. Skipping creation."
else
    echo "[2/4] Creating container '${CONTAINER_NAME}'..."
    docker run -dit --gpus all --shm-size 16G \
        -p 6006:6006 -p 8888:8888 \
        -v "$PWD":/workspace -w /workspace \
        --name $CONTAINER_NAME \
        $IMAGE bash
fi

# 3. Install system and Python dependencies
echo "[3/4] Installing dependencies inside container..."
docker exec -it $CONTAINER_NAME bash -c "
    apt-get update && \
    apt-get install -y libgl1-mesa-dev libglib2.0-0 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
"

# 4. Drop into the container shell
echo "[4/4] Attaching to container '${CONTAINER_NAME}'..."
docker exec -it $CONTAINER_NAME bash
