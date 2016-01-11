import numpy as np
import pysam
import argparse
import os, pdb

MIN_MAP_QUAL = 10

def parse_args():
    parser = argparse.ArgumentParser(description=" convert bam data format to bigWig data format, "
                                     " for ribosome profiling and RNA-seq data ")

    parser.add_argument("--dtype",
                        choices=("rnaseq","riboseq"),
                        default="riboseq",
                        help="specifies the type of assay (default: riboseq)")

    parser.add_argument("bam_file",
                        action="store",
                        help="path to bam input file")

    options = parser.parse_args()

    return options

def convert_rnaseq(options):

    # file names and handles
    count_file = os.path.splitext(options.bam_file)[0]
    sam_handle = pysam.Samfile(options.bam_file, "rb")
    count_handle = open(count_file, 'w')

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize count array
        counts = dict()
        for read in sam_iter:

            # skip read if unmapped
            if read.is_unmapped:
                continue

            # skip read, if mapping quality is low
            if read.mapq < MIN_MAP_QUAL:
                continue

            if read.is_reverse:
                site = read.pos + read.alen - 1
            else:
                site = read.pos

            try:
                counts[site] += 1
            except KeyError:
                counts[site] = 1

        # write counts to output file
        indices = np.sort(counts.keys())
        for i in indices:
            count_handle.write('\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%counts[i]])+'\n')

        print "completed %s"%cname

    sam_handle.close()
    count_handle.close()

    # index count file
    pysam.tabix_compress(count_file, count_file+'.gz', force=True)
    bgz_file = pysam.tabix_index(count_file+'.gz', force=True, zerobased=True, seq_col=1, start_col=2, end_col=3)
    print "Compressed file with RNA-seq counts is %s"%bgz_file


def convert_riboseq(options):

    # file names and handles
    fwd_count_file = os.path.splitext(options.bam_file)[0]+'_fwd'
    rev_count_file = os.path.splitext(options.bam_file)[0]+'_rev'
    sam_handle = pysam.Samfile(options.bam_file, "rb")
    fwd_count_handle = open(fwd_count_file, 'w')
    rev_count_handle = open(rev_count_file, 'w')

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize count arrays
        fwd_counts = dict()
        rev_counts = dict()
        for read in sam_iter:

            # skip read if unmapped
            if read.is_unmapped:
                continue

            # skip read, if mapping quality is low
            if read.mapq < MIN_MAP_QUAL:
                continue
            
            if read.is_reverse:
                asite = int(read.positions[-11])
                try:
                    rev_counts[asite] += 1
                except KeyError:
                    rev_counts[asite] = 1
            else:
                asite = int(read.positions[11])
                try:
                    fwd_counts[asite] += 1
                except KeyError:
                    fwd_counts[asite] = 1

        # write counts to output files
        indices = np.sort(fwd_counts.keys())
        for i in indices:
            fwd_count_handle.write('\t'.join([cname, '%d'%i, '%d'%(i+1), '%d'%fwd_counts[i]])+'\n')

        indices = np.sort(rev_counts.keys())
        for i in indices:
            rev_count_handle.write('\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%rev_counts[i]])+'\n')

        print "completed %s"%cname

    sam_handle.close()
    fwd_count_handle.close()
    rev_count_handle.close()

    # index count file
    pysam.tabix_compress(fwd_count_file, fwd_count_file+'.gz', force=True)
    bgz_file = pysam.tabix_index(fwd_count_file+'.gz', force=True, zerobased=True, seq_col=1, start_col=2, end_col=3)
    print "Compressed file with ribosome footprint counts on forward strand is %s"%bgz_file
    pysam.tabix_compress(rev_count_file, rev_count_file+'.gz', force=True)
    bgz_file = pysam.tabix_index(rev_count_file+'.gz', force=True, zerobased=True, seq_col=1, start_col=2, end_col=3)
    print "Compressed file with ribosome footprint counts on reverse strand is %s"%bgz_file

if __name__=="__main__":

    options = parse_args()

    if options.dtype=="rnaseq":
        convert_rnaseq(options)

    elif options.dtype=="riboseq":
        convert_riboseq(options)

