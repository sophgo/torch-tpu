#!/bin/bash

for fn in ${@}; do
    echo "#include \"${fn}\""
done
