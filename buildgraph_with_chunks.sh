#!/usr/bin/bash

POSITIONAL=()
argc=0
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -r|--reads)
      reads="$2"
      argc=$((++argc))
      shift # past argument
      shift # past value
      ;;
    -c|--chunksize)
      chunksize="$2"
      argc=$((++argc))
      shift # past argument
      shift # past value
      ;;
    -o|--output)
      exp="$2"
      argc=$((++argc))
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ ! $argc -eq 3 ]
then
    echo "USAGE: -r|--reads <READS> -c|--chunksize <CHUNK_SIZE> -o|--output <OUTDIR>"
    exit -1
fi

[ ! -d $exp ] && mkdir $exp
[ -d $exp/"chunked_reads" ] && rm -rf $exp/chunked_reads

$(dirname $0)/split $reads $chunksize $exp/chunked_reads

i=1
for chunk in $exp/chunked_reads/*.chunk.fasta
do
echo "Processing chunk " $i
kbm2 -q  -i $reads -d $chunk -n 2000 -l 2560 -t 32 | python $(dirname $0)/filter_alignments.py $exp/chunked_reads/ $i.
# kbm2 -q  -i $reads -d $chunk -n 2000 -l 1024 -t 32 | python $(dirname $0)/filter_alignments.py $exp/chunked_reads/ $i.
i=$((++i))
done;

echo "Combining subgraphs"
python $(dirname $0)/reduce.py $exp

echo "Cleaning chunks"
rm -rf $exp/chunked_reads