#!/bin/bash
# run.sh

# Cleaning on exit
cleanup() {
    echo "Cleaning up..."
    docker rm -f 3d-scanner-container 2>/dev/null
}
trap cleanup EXIT

# Determine OS type
OS_TYPE=$(uname)

# Get system resources based on OS
if [ "$OS_TYPE" = "Darwin" ]; then
    # macOS
    AVAILABLE_MEMORY=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    AVAILABLE_CPUS=$(sysctl -n hw.ncpu)
else
    # Linux
    AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    AVAILABLE_CPUS=$(nproc)
fi

# Set reasonable defaults if detection fails
if [ -z "$AVAILABLE_MEMORY" ] || [ "$AVAILABLE_MEMORY" -eq 0 ]; then
    AVAILABLE_MEMORY=16
fi
if [ -z "$AVAILABLE_CPUS" ] || [ "$AVAILABLE_CPUS" -eq 0 ]; then
    AVAILABLE_CPUS=4
fi

DOCKER_MEMORY=$((AVAILABLE_MEMORY * 95 / 100)) # use 80% available memory
DOCKER_MEMORY_SWAP=$((DOCKER_MEMORY * 6)) # swap is twice of main memory
USE_CPUS=$((AVAILABLE_CPUS > 4 ? 4 : AVAILABLE_CPUS)) # use max 4

echo "System resources:"
echo "OS Type: ${OS_TYPE}"
echo "Available memory: ${AVAILABLE_MEMORY}GB"
echo "Docker memory limit: ${DOCKER_MEMORY}GB"
echo "CPU cores to use: ${USE_CPUS}"

# Check minimal requirements
if [ ${AVAILABLE_MEMORY} -lt 8 ]; then
    echo "Error: Minimum 8GB of RAM required"
    exit 1
fi

echo "Building Docker image..."
docker build -t 3d-scanner .

# Run with error handling
docker run --rm \
    --name 3d-scanner-container \
    --memory="${DOCKER_MEMORY}g" \
    --memory-swap="${DOCKER_MEMORY_SWAP}g" \
    --cpus="${USE_CPUS}" \
    --shm-size="${DOCKER_MEMORY}g" \
    -v "$(pwd)/images:/app/images:rw" \
    -v "$(pwd)/output:/app/output:rw" \
    -v "$(pwd)/database:/app/database:rw" \
    --tmpfs /tmp:exec,size=${DOCKER_MEMORY}G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e OPENMVS_MAX_MEMORY=$((DOCKER_MEMORY * 1024)) \
    -e OMP_NUM_THREADS=${USE_CPUS} \
    -e COLMAP_NUM_THREADS=${USE_CPUS} \
    3d-scanner python3 /app/src/create_3d_model.py "$@"

# Checking return code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Container exited with error code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Processing completed successfully"