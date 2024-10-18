FROM sophgo/tpuc_dev:v3.1-base

WORKDIR /workspace
COPY requirements.txt ./
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm -rf ~/.cache/pip/* requirements.txt
