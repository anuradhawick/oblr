import sys
import fileinput
import gzip
import numpy as np

path = "./"
prefix = ""

if len(sys.argv) >= 2:
    path = sys.argv[1] + "/"

if len(sys.argv) >= 3:
    prefix = sys.argv[2]

class Alignment:
    def __init__(self, line):
        """
        COL1 qry_name
        COL2 qry_strand
        COL3 qry_length
        COL4 qry_beg
        COL5 qry_end
        COL6 ref_name
        COL7 ref_strand (always equals +)
        COL8 ref_length
        COL9 ref_beg
        COL10 ref_end
        COL11 match_len (length of matched k-mers)
        COL12 align_len (length of aligned)
        COL13 #kcnt (number of matched k-mers)
        COL14 #gap (number of gapped BINs)
        COL15 cigar (256 x SAM's cigar)
        """
        data = line.strip().split("\t")
        self.raw_data = line.strip()
        self.qry_name = data[0]
        self.qry_strand = data[1]
        self.qry_length = int(data[2])
        self.qry_beg = int(data[3])
        self.qry_end = int(data[4])
        self.ref_name = data[5]
        self.ref_strand = data[6]
        self.ref_length = int(data[7])
        self.ref_beg = int(data[8])
        self.ref_end = int(data[9])
        self.match_len = int(data[10])
        self.align_len = int(data[11])
        self.kmers = int(data[12])
        self.gap = int(data[13])
        

def is_overlap(alignment):
    qry_beg = alignment.qry_beg
    qry_end = alignment.qry_end
    ref_beg = alignment.ref_beg
    ref_end = alignment.ref_end
    qry_length = alignment.qry_length
    ref_length = alignment.ref_length

    THRESHOLD = 512

    # full overlap
    if qry_beg <= THRESHOLD and qry_length - qry_end <= THRESHOLD:
        return True
    elif ref_beg <= THRESHOLD and ref_length - ref_end <= THRESHOLD:
        return True

    # qry end overlap
    if qry_length - qry_end <= THRESHOLD and ref_beg <= THRESHOLD:
        return True
    # ref end overlap 
    elif ref_length - ref_end <= THRESHOLD and qry_beg <= THRESHOLD:
        return True

    return False

# import seaborn as sns
# import matplotlib.pyplot as plt

def process_batch(alignments, fpe, fpd):
    # skip alignments that are self, this can cause total failure
    # skip non overlaps
    alignments = [a for a in alignments if is_overlap(a) and a.qry_name!=a.ref_name]
    # exit if empty (first scenario)
    if len(alignments) == 0:
        return
    
    # compute alignment overlaps
    alignments = sorted(alignments, key=lambda a: a.match_len, reverse=True)
    match_lengths = [a.match_len for a in alignments]
    mean_match = np.mean(match_lengths)

    # print(alignments[0].raw_data)
    # print(alignments[0].qry_name)
    # print(match_lengths)
    # print()
    # sns.histplot(match_lengths)
    # plt.show()
    degree = 0
    for n, a in enumerate(alignments):    
        # if less than half of mean probably not a good match
        # if a.match_len < mean_match * 0.5:
        #     break

        # record actual edge count
        degree += 1

        # write only top 20 edges
        if n < 20:
            fpe.write(f"{a.qry_name}\t{a.ref_name}\n")

    fpd.write(f"{alignments[0].qry_name}\t{degree}\n")

active_query = None
alns_buffer = []
out_file_edges = open(path + prefix + 'reads.alns', 'w+')
out_file_degree = open(path + prefix + 'degree', 'w+')

for line in fileinput.input('-'):
    if len(line.strip()) == 1:
        continue

    alignment = Alignment(line)

    if alignment.qry_name != active_query:
        # new query
        # if there is a previous query process it
        if len(alns_buffer) > 0:
            process_batch(alns_buffer, out_file_edges, out_file_degree)
            # sys.exit(0)

        # reset buffers
        active_query = alignment.qry_name
        alns_buffer = [alignment]
    else:
        alns_buffer.append(alignment)

if len(alns_buffer) > 0:
    process_batch(alns_buffer, out_file_edges, out_file_degree)

out_file_edges.close()
out_file_degree.close()