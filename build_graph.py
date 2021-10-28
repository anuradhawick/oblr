#!/usr/bin/python3
import argparse
from tqdm import tqdm
import pandas as pd
import gzip
import os
import numpy as np
from collections import Counter

'''
GPU TIME
--------
User time (seconds): 65.21
System time (seconds): 18.72
Percent of CPU this job got: 92%
Elapsed (wall clock) time (h:mm:ss or m:ss): 1:30.83
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 4958320
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 18
Minor (reclaiming a frame) page faults: 11685454
Voluntary context switches: 3309
Involuntary context switches: 21635
Swaps: 0
File system inputs: 0
File system outputs: 1234944
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0

CPU TIME
--------
User time (seconds): 27363.27
System time (seconds): 907.53
Percent of CPU this job got: 1375%
Elapsed (wall clock) time (h:mm:ss or m:ss): 34:15.96
Average shared text size (kbytes): 0
Average unshared data size (kbytes): 0
Average stack size (kbytes): 0
Average total size (kbytes): 0
Maximum resident set size (kbytes): 3553416
Average resident set size (kbytes): 0
Major (requiring I/O) page faults: 283
Minor (reclaiming a frame) page faults: 7444090
Voluntary context switches: 13964018
Involuntary context switches: 3042188
Swaps: 0
File system inputs: 54928
File system outputs: 1234952
Socket messages sent: 0
Socket messages received: 0
Signals delivered: 0
Page size (bytes): 4096
Exit status: 0
'''

try:
    from cuml.manifold import UMAP
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as silhouette_score
    from cuml.cluster import HDBSCAN
    cuda = True
except Exception as e:
    from sklearn.metrics import silhouette_score
    from umap import UMAP
    from hdbscan import HDBSCAN
    print('Using SciPy', e)
    cuda = False


def get_idx_maps(read_ids_file_path, truth):
    reads_truth = {}
    read_id_idx = {}

    with open(read_ids_file_path) as read_ids_file:
        for t, rid in tqdm(zip(truth, read_ids_file)):
            rid = rid.strip()[1:]
            reads_truth[rid] = t
            read_id_idx[rid] = len(read_id_idx)

    return reads_truth, read_id_idx


def load_read_degrees(degrees_file_path, size, read_id_idx):
    degree_array = np.zeros(size, dtype=int)

    for line in tqdm(open(degrees_file_path, 'r')):
        i, d = line.strip().split()
        d = int(d)
        degree_array[read_id_idx[i]] = d

    return degree_array


def alignments_to_edges(alignments_file_path, edges_txt_path, read_id_idx, reads_truth):
    TP = 0
    FP = 0

    if not os.path.isfile(edges_txt_path):
        with open(edges_txt_path, "w+") as ef:
            for line in tqdm(open(alignments_file_path, "r")):
                u, v = line.strip().split('\t')

                if u == v:
                    continue

                ef.write(f"{read_id_idx[u]}\t{read_id_idx[v]}\n")

                if reads_truth[u] == 'Unknown' or reads_truth[v] == 'Unknown':
                    continue
                if reads_truth[u] == reads_truth[v]:
                    TP += 1
                else:
                    FP += 1
    return TP, FP


def load_edges_as_numpy(edges_txt_path, edges_npy_path):
    if not os.path.isfile(edges_npy_path):
        edges_txt = [x.strip() for x in tqdm(open(edges_txt_path))]
        edges = np.zeros((len(edges_txt), 2), dtype=np.int32)

        for i in tqdm(range(len(edges_txt))):
            e1, e2 = edges_txt[i].strip().split()
            edges[i] = [int(e1), int(e2)]

        np.save(edges_npy_path, edges)

    return np.load(edges_npy_path)


def get_highest_scoring_clustering(data, size):
    best_score = -1
    best_clusters = None
    best_size = -1
    
    try:
        clusters = HDBSCAN(min_cluster_size=size).fit_predict(data)
        if len(set(clusters) - {1}) == 0:
            return None, -1, None
        score = silhouette_score(data[clusters!=-1], clusters[clusters!=-1])

        print(size, score)
    except:
        pass

    if score > best_score:
        best_score = score
        best_clusters = clusters
        best_size = size
    
    return best_size, best_score, best_clusters


def get_best_embedding(data, weights):
    best_size = None
    best_sample_size = None
    best_sample_idx = None
    best_score = -1
    best_clusters = None
    best_embedding = None
    best_cluster_count = None
    
    for sample_size in [25000, 50000, 100000]:
        print(f'Scanning sample size {sample_size}')
        sample_idx = np.random.choice(range(len(data)), size=sample_size, replace=False, p=weights/weights.sum())
        sampled_data = data[sample_idx]
        embedding = UMAP().fit_transform(sampled_data)
        size, score, clusters = get_highest_scoring_clustering(embedding, 500)
        count = len(set(clusters) - {-1})
        
        print(f'Cluster size = {size:5} Clusters = {count:5} Score = {score:1.5f}')
        
        if score > best_score:
            best_cluster_count = count
            best_size = size
            best_sample_size = sample_size
            best_score = score
            best_clusters = clusters
            best_embedding = embedding
            best_sample_idx = sample_idx
            
    return best_size, best_sample_size, best_score, best_clusters, best_cluster_count, best_embedding, best_sample_idx


def rename_clusters(clusters):
    rename_map = {k: n for n, k in enumerate(set(clusters) - {-1})}
    rename_map[-1] = -1

    clusters = np.array([rename_map[x] for x in clusters])

    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""OBLR Build Graph Routine.""")

    parser.add_argument('--alignments', '-a',
                        help="Alignments file reads.alns.gz",
                        type=str,
                        required=True)
    parser.add_argument('--degree', '-d',
                        help="Read degree file degree.gz",
                        type=str,
                        required=True)
    parser.add_argument('--features', '-f',
                        help="K-mer features file",
                        type=str,
                        required=True)
    parser.add_argument('--read-ids', '-i',
                        help="Set of ids for the reads",
                        type=str,
                        required=True)
    parser.add_argument('--ground-truth', '-g',
                        help="Ground truth of reads for dry runs and sensitivity tuning",
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument(
        '--output', '-o', help="Output directory", type=str, required=True)
    args = parser.parse_args()

    alignments_file_path = args.alignments
    degrees_file_path = args.degree
    features = args.features
    has_truth = args.ground_truth is not None
    output = args.output
    read_ids = args.read_ids

    comp = pd.read_csv(features, delimiter=' ', header=None).to_numpy()

    if has_truth:
        truth = np.array(open(args.ground_truth).read().strip().split("\n"))
    else:
        truth = np.array([str(0) for x in range(len(comp))])
        has_truth = False

    if not os.path.isdir(output):
        os.mkdir(output)

    reads_truth, read_id_idx = get_idx_maps(read_ids, truth)

    degree_array = load_read_degrees(degrees_file_path, len(comp), read_id_idx)

    print(f"Total edges = {degree_array.sum()}")

    # loading edges
    TP, FP = alignments_to_edges(
        alignments_file_path, output + "/edges.txt", read_id_idx, reads_truth)
    edges = load_edges_as_numpy(output + "/edges.txt", output + "/edges.npy")

    # sampling weights logic
    sample_weights = np.zeros_like(degree_array, dtype=np.float32)
    sample_scale = np.ones_like(degree_array, dtype=np.float32)

    for n, d in enumerate(degree_array):
        sample_weights[n] = 1.0/d if d > 0 else 0
        sample_scale[n] = max(1, np.log10(max(1, d)))

    scaled = comp * sample_scale.reshape(-1, 1)

    # cluster discovery
    results = []

    for i in range(5):
        t_size, t_sample_size, t_score, t_clusters, t_cluster_count, t_embedding, t_sample_idx = get_best_embedding(
            scaled, sample_weights)

        results.append([t_size, t_sample_size, t_score, t_clusters,
                       t_cluster_count, t_embedding, t_sample_idx])

    cluster_counts = Counter([result[4] for result in results])
    cluster_counts = sorted(cluster_counts.most_common(),
                            key=lambda x: (x[1], x[0]), reverse=True)
    chose = cluster_counts[0][0]

    print(f"Count stats = {cluster_counts}")
    print(f'Maximally occuring cluster count = {chose}')

    size, sample_size, score, clusters, cluster_count, embedding, sample_idx = None, None, - \
        1, None, 0, None, None

    for result in results:
        if result[4] == chose and result[2] > score:
            size, sample_size, score, clusters, cluster_count, embedding, sample_idx = result

    print(
        f'Chosen Score = {score} Sample_size = {sample_size:5} Size = {size} Clusters = {cluster_count}')

    # rename clusters
    clusters = rename_clusters(clusters)

    # save the results
    read_cluster = np.array([[r, c]
                            for r, c in zip(sample_idx, clusters) if c != -1])
    np.savez(output + '/data.npz', edges=edges,
             scaled=scaled, read_cluster=read_cluster)
