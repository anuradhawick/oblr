#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <zlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "kseq.h"

KSEQ_INIT(gzFile, gzread)

using namespace std;

int main(int ac, char **av)
{
    if (ac != 4)
    {
        cout << "Usage stats <FASTQ/FASTA File> <CHUNK_SIZE> <OUT_DIR>" << endl;
        return -1;
    }

    string path = av[1];
    float chunksize = stof(av[2]);
    string outdir = av[3];

    struct stat info;

    if (info.st_mode & S_IFDIR)
    {
        cout << "Out directory already exist. Please remove " << outdir << endl;
        return -1;
    }
    else
    {
        mkdir(outdir.c_str(), 0755);
    }

    gzFile fp = gzopen(path.c_str(), "r");
    kseq_t *ks = kseq_init(fp);
    int ret;
    u_int64_t no_reads = 0;

    while ((ret = kseq_read(ks)) >= 0)
    {
        no_reads++;
    }
    cout << "Number of sequences " << no_reads << endl;

    gzrewind(fp);
    kseq_rewind(ks);

    float chunks = ceil((float)no_reads / chunksize);
    cout << "Breaking into " << chunks << " chunks" << endl;

    no_reads = 0;
    ofstream fso;
    int chunkid = 1;
    while ((ret = kseq_read(ks)) >= 0)
    {
        no_reads++;

        if (no_reads >1 && no_reads % (int)chunksize == 1)
        {
            chunkid++;
            fso.close();
        }

        if (!fso.is_open())
        {
            fso.open(outdir + "/" + to_string(chunkid) + ".chunk.fasta");
        }

        fso << ">" << ks->name.s << "\n" << ks->seq.s << "\n";
    }

    fso.close();
    gzclose(fp);
    kseq_destroy(ks);
}