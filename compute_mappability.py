import numpy as np
import subprocess
import pysam
import argparse
import os, pdb

MIN_MAP_QUAL = 10

def parse_args():
    parser = argparse.ArgumentParser(description=" convert bam data format to tabix data format, "
                                     " for ribosome profiling and RNA-seq data ")

    parser.add_argument("bam_file",
                        action="store",
                        help="path to bam input file")

    parser.add_argument("mappability_file_prefix",
                        action="store",
                        help="prefix of mappability file")

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

def compute_mappability(options):

    # file names and handles
    map_file = options.mappability_file_prefix
    map_handle = open(map_file, 'w')
    sam_handle = pysam.Samfile(options.bam_file, "rb")

    for cname,clen in zip(sam_handle.references,sam_handle.lengths):

        # fetch reads in chromosome
        sam_iter = sam_handle.fetch(reference=cname)

        # initialize mappable positions
        mappable_positions = []
        for read in sam_iter:

            # skip read if unmapped
            if read.is_unmapped:
                continue

            # skip read, if mapping quality is low
            if read.mapq < MIN_MAP_QUAL:
                continue

            if not read.is_reverse:
                mapped_site = int(read.positions[0])
                true_chrom, true_site = read.query_name.split(':')[:2]
                if read.reference_name==true_chrom and mapped_site==int(true_site):
                    mappable_positions.append(mapped_site)

        if len(mappable_positions)>0:

            # get boundaries of mappable portions of the genome
            mappable_positions = np.sort(mappable_positions)

            boundaries = mappable_positions[:-1]-mappable_positions[1:]
            indices = np.where(boundaries<-1)[0]
            ends = (mappable_positions[indices]+1).tolist()
            try:
                ends.append(mappable_positions[-1]+1)
            except IndexError:
                pdb.set_trace()

            boundaries = mappable_positions[1:]-mappable_positions[:-1]
            indices = np.where(boundaries>1)[0]+1
            starts = mappable_positions[indices].tolist()
            starts.insert(0,mappable_positions[0])

            # write to file
            for start,end in zip(starts,ends):
                map_handle.write('\t'.join([cname, '%d'%start, '%d'%end])+'\n')

        print "completed %s"%cname

    sam_handle.close()
    map_handle.close()

    # compress count file
    pipe = subprocess.Popen("%s -f %s"%(options.bgzip, map_file), \
        stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]

    # index count file
    pipe = subprocess.Popen("%s -f -b 2 -e 3 -0 %s.gz"%(options.tabix, map_file), \
        stdout=subprocess.PIPE, shell=True)
    stdout = pipe.communicate()[0]

    print "completed computing mappability from BAM file %s"%options.bam_file

if __name__=="__main__":

    options = parse_args()

    compute_mappability(options)
