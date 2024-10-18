#!/bin/bash

TGI_SERVER_TIMEOUT=600 # 10min
REQUEST_TIMEOUT=1200 # 20min
MAX_NEW_TOKENS=3

function start_tgi_container() {
    local image_id=$1
    local port=$2
    docker run -td --entrypoint bash -e USE_DUMMY_DATA=ON -e DEBUG_MODE=ON -p ${port}:80 ${image_id}
}

function clean_container() {
    local container_id="$1"
    docker inspect $container_id &> /dev/null
    if [ $? -eq 0 ]; then
        local container_status=$(docker inspect --format='{{.State.Status}}' $container_id)
        if [ "$container_status" = "running" ]; then
            docker stop "$container_id" > /dev/null
        fi
        docker rm "$container_id" > /dev/null
        echo "Container $container_id is stoped and removed."
    else
        echo "Container $container_id does not exist, nothing to do."
    fi
}

function clean_image() {
    local image_id="$1"
    local containers=$(docker ps -a -q --filter "ancestor=$image_id")
    if [ -n "$containers" ]; then
        docker stop $containers
        docker rm $containers
    else
        echo "No containers found for image $image_id."
    fi
    docker image rm $image_id
}

function post_and_check() {
    local port="$1"
    local question="$2"
    local answer="$3"
    info=$(curl 127.0.0.1:${port}/generate --max-time ${REQUEST_TIMEOUT} -X POST -d "{\"inputs\":\"${question}\",\"parameters\":{\"max_new_tokens\":${MAX_NEW_TOKENS}}}" -H 'Content-Type: application/json')
    res=$?
    echo "Question: $question"
    echo "Result: $info"
    if [ $res -eq 0 ]; then
        echo "CHECK SUCCESS!"
    else
        echo "CHECK FAILED!"
    fi
    return $?
}

# get avaliable port between 8090-9090
function get_availed_port() {
    for port in $(seq 8090 9090); do
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    return 1
}

function wait_tgi_ready() {
    local container_id=$1
    if [[ -z "$container_id" ]]; then
        echo "container id should be specified."
        return 1
    fi

    for ((attempt=1; attempt<=${TGI_SERVER_TIMEOUT}; attempt++)); do
        if docker exec "$container_id" bash -c 'cat run.log' | grep -q 'Connected'; then
            echo "[INFO] TGI server ready."
            return 0
        fi
        sleep 1
    done
    docker exec "$container_id" bash -c 'cat run.log'
    echo "[ERROR] TGI server start failed."

    return 1
}

function tgi_update_torch_tpu() {
    CONTAINER_ID=$1
    TORCH_TPU_WHL=$2
    docker cp $TORCH_TPU_WHL $CONTAINER_ID:/opt/
    docker exec $CONTAINER_ID bash -c 'pip install --force-reinstall /opt/torch_tpu-*.whl'
}

function tgi_launch_server() {
    CONTAINER_ID=$1
    docker exec -td $CONTAINER_ID bash -c 'server/soph_entrypoint.sh --model-id /data/llama-2-7b-chat-hf >run.log 2>&1'
}

function tgi_regression() {
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    WORK_DIR=$(realpath "${CURRENT_DIR}/../../tgi-workspace")
    echo "WORKDIR: ${WORK_DIR}"

    # unzip torch_tpu and get whl
    local TORCHTPU_TAR=$(find ${WORK_DIR} -name torch-tpu*.gz | xargs ls -t | head -n 1)
    if [ -z "$TORCHTPU_TAR" ]; then
        echo "[ERROR] torch-tpu release package can not found!"
        return -1
    else
        echo "[INFO] torch-tpu release package found: $TORCHTPU_TAR"
    fi
    tar -xzvf $TORCHTPU_TAR -C $WORK_DIR
    local TORCH_TPU_WHL=$(find ${WORK_DIR}/dist/ -name torch_tpu*.whl)

    # find latest tgi docker
    local DOCKER_FILE=$(find ${WORK_DIR} -name docker-soph_tgi-*.bz2 | xargs ls -t | head -n 1)
    if [ -z "$DOCKER_FILE" ]; then
        echo "[ERROR] TGI docker image file can not found!"
        return -1
    else
        echo "[INFO] LATEST TGI DOCKER FILE FOUND: $DOCKER_FILE"
    fi

    # load image and get image id
    local IMAGE_ID=$(bunzip2 -c "$DOCKER_FILE" | docker load | grep 'Loaded image' | awk '{ print $3 }')
    echo "[INFO] DOCKER IMAGE ID: $IMAGE_ID"

    # get avaliable port
    PORT=$(get_availed_port)
    echo "[INFO] PORT FOR TGI SERVER: $PORT"

    # start TGI server
    CONTAINER_ID=$(start_tgi_container "$IMAGE_ID" "$PORT")
    echo "[INFO] TGI server started at container: ${CONTAINER_ID}"

    # update torch_tpu
    tgi_update_torch_tpu "$CONTAINER_ID" "$TORCH_TPU_WHL"
    echo "[INFO] torch_tpu updated success by: ${TORCH_TPU_WHL}"

    #launch tgi server
    tgi_launch_server "$CONTAINER_ID"
    wait_tgi_ready "$CONTAINER_ID"

    # post req and check answer
    question="What is Deep Learning?"
    answer="\n\nDeep learning is a subset of machine learning that involves the use of artificial neural networks to"
    post_and_check "$PORT" "$question" "$answer"
    res=$?

    # clean
    clean_container "$CONTAINER_ID"
    clean_image "$IMAGE_ID"
    rm -rf $WORK_DIR/dist
    #rm -rf $DOCKER_FILE

    return $res
}



