py_verison=${1:-py310} 
docker build -f tpuc_dev_base_${py_verison}.Dockerfile   -t sophgo/tpuc_dev:v3.1-base               .
docker build -f tpuc_dev_whl.Dockerfile                  -t sophgo/torch_tpu:v0.1-${py_verison}     .