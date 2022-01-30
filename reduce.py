from glob import glob
import sys
from collections import defaultdict
import random

read_degree = defaultdict(int)
read_alns = defaultdict(set)

exp_path = sys.argv[1]



for degreef in glob(exp_path + "./chunked_reads/*.degree"):
    alnsf = degreef.replace("degree", "reads.alns")
    
    with open(degreef) as f:
        for line in f:
            rid, degree = line.strip().split()
            degree = int(degree)

            read_degree[rid] += degree

    with open(alnsf) as f:
        for line in f:
            r1, r2 = line.strip().split()
            if r1 not in read_alns or len(read_alns[r1]) < 100:
                read_alns[r1].add(r2)
 
with open(exp_path + "/degree", "w+") as degreeallf:
    for k, v in read_degree.items():
        degreeallf.write(f"{k}\t{v}\n")

with open(exp_path + "/reads.alns", "w+") as alnsallf:
    for r1 in read_alns.keys():
        r2_list = list(read_alns[r1])
        r2_list = random.sample(r2_list, min(50, len(r2_list)))

        for r2 in r2_list:
            alnsallf.write(f"{r1}\t{r2}\n")