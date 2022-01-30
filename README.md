# Read Over Binning of Long Reads (OBLR)

This is a pipeline to perform long reads binning using the read overlap information.

## Pre-requisites

### Tools needed
1. Install **seq2vec** from [https://github.com/anuradhawick/seq2vec](https://github.com/anuradhawick/seq2vec). Please add the binary to your PATH variable.
2. Install **wtdbg2** (which hosts kbm2) from [https://github.com/ruanjue/wtdbg2](https://github.com/ruanjue/wtdbg2). Please add the binary to your PATH variable.
3. Install **seqtk** from [https://github.com/lh3/seqtk](https://github.com/lh3/seqtk)

### Python dependencies

You can run OBLR with CPU only. However, this can be extremely slow. The recommended approach is to have two python environments; 1) for pytorch 2) rapids.ai. Currently they are pre-built on different CUDA-toolkits. In future, one environment would be sufficient.

#### PyTorch.org environment should have the following

* biopython
* pytorch
* pytorch geometric
* numpy
* tqdm
* matplotlib (only for notebooks with plots)
* seaborn (only for notebooks with plots)

#### Rapids.AI environment should have the following

* CuML from (https://rapids.ai/start.html)[https://rapids.ai/start.html]
* tqdm
* numpy
* pandas
* matplotlib (only for notebooks with plots)
* seaborn (only for notebooks with plots)

> Note: If you're planning to skip Rapids.AI installation please make sure you have **umap-learn** and **hdbscan** installed in the pytorch environment. Please install **hdbscan** from github as conda bundle may have bugs. 

## How to run with Jupyter Notebooks

On one terminal tab, run following.
```
conda activate rapids-21.10
jupyter notebook --port 8888
```
In another terminal tab, run following.
```
conda activate pytorch
jupyter notebook --port 8889
```
> You can indeed run with nohup in background.

Now open `step-1-4.ipynb` file from `rapids-21.10` environment. Follow the instructions inside the notebook. Once finished, open `step-5.ipynb` from the `pytorch` environment and follow instruction. At the end of the notebook you can separate reads into bins for assembly task.

## How to run in a server

One might consider running the program on a server. The pipeline is currently availabe as scripts.

**Step-1:** Preprocess reads

```bash
# your experiment path which has reads
exp = "./test_data/"

# renaming reads
seqtk rename $exp/reads.fastq read_ | seqtk seq -A > $exp/reads.fasta

# obtaining read ids
grep ">" $exp/reads.fasta > $exp/read_ids
```

**Step-2:** Build the graph

```bash
exp = "./test_data/"

# compute 4mer vectors (-t for threads)
seq2vec -k 4 -o $exp/4mers -f $exp/reads.fasta -t 32

# build the graph using chunked reads
./buildgraph_with_chunks.sh -r $exp//reads.fasta -c <CHUNK_SIZE> -o $exp/
```

**Step-3:** Detect clusters
```bash
exp = "./test_data/"

# activate rapids environment (or use pytorch environment as advised)
conda activate rapids-21.10

# reads.alns and degree are created from kbm2 pipeline command
python ./build_graph.py \
            -a $exp/reads.alns \
            -d $exp/degree \
            -f $exp/4mers \
            -i $exp/read_ids \
            -o $exp/
```

**Step-4:** Detect clusters
```bash
exp = "./test_data/"

# activate pytorch environment
conda activate pytorch

# data.npz used from step-2
python sage_label_prop.py \
            -d $exp/data.npz \
            -o $exp/
```

> Note if you chose to run everything in a single script, refer to file `oblr_runner.sh` to see how one can change conda environment in a bash file.

### :stop: 

This code is under construction. Hence notebooks are advised over the pipeline. It also facilitates plots and much more. `kbm2` is the most resource demanding step. ~32GB of RAM of above is recommended. If you have fast storage consider increasing swap. May be slower but kbm2 will run for sure.

Rapids.AI is advise as worst case GPU time was 1:30.83 while CPU time was 34:15.96 in a 32 thread machine (almost 1000x gain in speed compared to single threaded mode).