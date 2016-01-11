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
                        choices=("RNAseq","Riboseq"),
                        default="Riboseq",
                        help="specifies the type of assay (default: Riboseq)")

    parser.add_argument("bam_file",
                        action=store,
                        help="path to bam input file")

    options = parser.parse_args()

    return options

def convert_rnaseq(options):

    # file names and handles
    wigfile = os.path.splitext(options.bam_file)[0]+'.wig'
    sam_handle = pysam.Alignmentfile(options.bam_file, "rb")
    wig_handle = open(options.wig_file, 'w')

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
                site = read.pos + read.alen
            else:
                site = read.pos + 1

            counts[site] += 1

        # write counts to wig file
        wig_handle.write('variableStep chrom=%s\n'%cname)
        indices = np.where(counts!=0)[0]
        towrite = '\n'.join([' '.join(['%d'%i,'%d'%c]) for i,c in zip(indices,counts[indices])])
        wig_handle.write(towrite)

        print "completed %s"%cname

    sam_handle.close()
    wig_handle.close()


def convert_riboseq(options):

    # file names and handles
    fwd_wig_file = os.path.splitext(options.bam_file)[0]+'_fwd.wig'
    rev_wig_file = os.path.splitext(options.bam_file)[0]+'_rev.wig'
    sam_handle = pysam.Alignmentfile(options.bam_file, "rb")
    fwd_wig_handle = open(fwd_wig_file, 'w')
    rev_wig_handle = open(rev_wig_file, 'w')

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

        # write counts to wig files
        fwd_wig_handle.write('variableStep chrom=%s\n'%cname)
        indices = np.where(fwd_counts!=0)[0]
        towrite = '\n'.join([' '.join(['%d'%i,'%d'%c]) for i,c in zip(indices,fwd_counts[indices])])
        fwd_wig_handle.write(towrite)

        rev_wig_handle.write('variableStep chrom=%s\n'%cname)
        indices = np.where(rev_counts!=0)[0]
        towrite = '\n'.join([' '.join(['%d'%i,'%d'%c]) for i,c in zip(indices,rev_counts[indices])])
        rev_wig_handle.write(towrite)

        print "completed %s"%cname

    sam_handle.close()
    fwd_wig_handle.close()
    rev_wig_handle.close()

if __name__=="__main__":

    options = parse_args()

    if options.dtype=="RNAseq":
        convert_rnaseq(options)

    elif options.dtype=="Riboseq":
        convert_riboseq(options)

