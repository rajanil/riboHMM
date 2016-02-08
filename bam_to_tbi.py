import numpy as np
import pysam
import subprocess
import argparse
import utils
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

    options.bgzip = which("bgzip")
    options.tabix = which("tabix")

    return options

def which(program):

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

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

    # compress count file
    pipe = subprocess.Popen("%s -f %s"%(options.bgzip, count_file), \
        stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]

    # index count file
    pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.gz"%(options.tabix, count_file), \
        stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]
    print "Compressed file with RNA-seq counts is %s"%(count_file+'.gz')


def convert_riboseq(options):

    # file names and handles
    fwd_count_file = os.path.splitext(options.bam_file)[0]+'_fwd'
    rev_count_file = os.path.splitext(options.bam_file)[0]+'_rev'
    sam_handle = pysam.Samfile(options.bam_file, "rb")
    fwd_handle = dict([(r,open(fwd_count_file+'.%d'%r, 'w')) for r in utils.READ_LENGTHS])
    rev_handle = dict([(r,open(rev_count_file+'.%d'%r, 'w')) for r in utils.READ_LENGTHS])

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize count arrays
        fwd_counts = dict([(r,dict()) for r in utils.READ_LENGTHS])
        rev_counts = dict([(r,dict()) for r in utils.READ_LENGTHS])
        for read in sam_iter:

            # skip reads not of the appropriate length
            if read.rlen not in utils.READ_LENGTHS:
                continue

            # skip read if unmapped
            if read.is_unmapped:
                continue

            # skip read, if mapping quality is low
            if read.mapq < MIN_MAP_QUAL:
                continue
            
            if read.is_reverse:
                asite = int(read.positions[-13])
                try:
                    rev_counts[read.rlen][asite] += 1
                except KeyError:
                    rev_counts[read.rlen][asite] = 1
            else:
                asite = int(read.positions[12])
                try:
                    fwd_counts[read.rlen][asite] += 1
                except KeyError:
                    fwd_counts[read.rlen][asite] = 1

        # write counts to output files
        for r in utils.READ_LENGTHS:
            indices = np.sort(fwd_counts[r].keys())
            for i in indices:
                fwd_handle[r].write('\t'.join([cname, '%d'%i, '%d'%(i+1), '%d'%fwd_counts[r][i]])+'\n')

            indices = np.sort(rev_counts[r].keys())
            for i in indices:
                rev_handle[r].write('\t'.join([cname, '%d'%i, '%d'%(i+1), '%d'%rev_counts[r][i]])+'\n')

        print "completed %s"%cname

    sam_handle.close()
    for r in utils.READ_LENGTHS:
        fwd_handle[r].close()
        rev_handle[r].close()

    for r in utils.READ_LENGTHS:

        # compress count file
        pipe = subprocess.Popen("%s -f %s.%d"%(options.bgzip, fwd_count_file, r), \
            stdout=subprocess.PIPE, shell=True)
        stdout = pipe.communicate()[0]
        pipe = subprocess.Popen("%s -f %s.%d"%(options.bgzip, rev_count_file, r), \
            stdout=subprocess.PIPE, shell=True)
        stdout = pipe.communicate()[0]

        # index count file
        pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.%d.gz"%(options.tabix, fwd_count_file, r), \
            stdout=subprocess.PIPE, shell=True)
        stdout = pipe.communicate()[0]
        pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.%d.gz"%(options.tabix, rev_count_file, r), \
            stdout=subprocess.PIPE, shell=True)
        stdout = pipe.communicate()[0]
        print "Compressed file with ribosome footprint counts on forward strand is %s"%(fwd_count_file+'.%d.gz'%r)
        print "Compressed file with ribosome footprint counts on reverse strand is %s"%(rev_count_file+'.%d.gz'%r)

if __name__=="__main__":

    options = parse_args()

    if options.dtype=="rnaseq":
        convert_rnaseq(options)

    elif options.dtype=="riboseq":
        convert_riboseq(options)

