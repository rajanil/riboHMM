import numpy as np
import pysam
import argparse
import utils
import os

MIN_MAP_QUAL = 10

def parse_args():
    parser = argparse.ArgumentParser(description=" convert bam data format to bigWig data format, "
                                     " for ribosome profiling and RNA-seq data ")

    parser.add_argument("--dtype",
                        choices=("rnaseq","riboseq"),
                        default="riboseq",
                        help="specifies the type of assay (default: riboseq)")

    parser.add_argument("bam_file",
                        action=store,
                        help="path to bam input file")

    options = parser.parse_args()

    return options

def convert_rnaseq(options):

    # file names and handles
    count_file = os.path.splitext(options.bam_file)[0]
    sam_handle = pysam.Alignmentfile(options.bam_file, "rb")
    count_handle = open(count_file, 'w')

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize count array
        counts = np.array((clen,), dtype='int')
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

            counts[site] += 1

        # write counts to output file
        indices = np.where(counts!=0)[0]
        towrite = '\n'.join(['\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%c]) for i,c in zip(indices,counts[indices])])
        count_handle.write(towrite)

        print "completed %s"%cname

    sam_handle.close()
    count_handle.close()

    # index count file
    bgz_file = pysam.tabix_index(count_file, force=True, zerobased=True)

    print "Compressed file with RNA-seq counts is %s"%bgz_file


def convert_riboseq(options):

    # file names and handles
    fwd_count_file = os.path.splitext(options.bam_file)[0]+'_fwd'
    rev_count_file = os.path.splitext(options.bam_file)[0]+'_rev'
    sam_handle = pysam.Alignmentfile(options.bam_file, "rb")
    fwd_count_handle = open(fwd_count_file, 'w')
    rev_count_handle = open(rev_count_file, 'w')

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize count arrays
        fwd_counts = np.array((clen,), dtype='int')
        rev_counts = np.array((clen,), dtype='int')
        for read in sam_iter:

            # skip read if unmapped
            if read.is_unmapped:
                continue

            # skip read, if mapping quality is low
            if read.mapq < MIN_MAP_QUAL:
                continue
            
            if read.is_reverse:
                asite = read.positions[-11]
                rev_counts[asite] += 1
            else:
                asite = read.positions[11]
                fwd_counts[asite] += 1

        # write counts to output files
        indices = np.where(fwd_counts!=0)[0]
        towrite = '\n'.join(['\t'.join([cname,'%d'%i,'%d'%(i+1),'%d'%c]) for i,c in zip(indices,fwd_counts[indices])])
        fwd_count_handle.write(towrite)

        indices = np.where(rev_counts!=0)[0]
        towrite = '\n'.join([' '.join([cname,'%d'%i,'%d'%(i+1),'%d'%c]) for i,c in zip(indices,rev_counts[indices])])
        rev_count_handle.write(towrite)

        print "completed %s"%cname

    sam_handle.close()

    # index count file
    bgz_file = pysam.tabix_index(fwd_count_file, force=True, zerobased=True)
    print "Compressed file with ribosome footprint counts on forward strand is %s"%bgz_file
    bgz_file = pysam.tabix_index(rev_count_file, force=True, zerobased=True)
    print "Compressed file with ribosome footprint counts on reverse strand is %s"%bgz_file

if __name__=="__main__":

    options = parse_args()

    if options.dtype=="rnaseq":
        convert_rnaseq(options)

    elif options.dtype=="riboseq":
        convert_riboseq(options)

