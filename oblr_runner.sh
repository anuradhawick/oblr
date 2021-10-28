#!/usr/bin/bash

# set 1
exp=./test_data/
rapids_env=rapids-21.10
python_env=py37

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $rapids_env

python ./build_graph.py \
            -a $exp/reads.alns \
            -d $exp/degree \
            -f $exp/4mers \
            -i $exp/read_ids \
            -o $exp/


source $(conda info --base)/etc/profile.d/conda.sh
conda activate $python_env

python sage_label_prop.py \
            -d $exp/data.npz \
            -o $exp/

