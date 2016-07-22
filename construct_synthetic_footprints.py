import argparse
import warnings
import gzip
import pdb
import os

import numpy as np
import pysam

import load_data
import utils

# ignore warnings with these expressions
warnings.filterwarnings('ignore', '.*overflow encountered.*',)
warnings.filterwarnings('ignore', '.*divide by zero.*',)
warnings.filterwarnings('ignore', '.*invalid value.*',)

def parse_args():
    parser = argparse.ArgumentParser(description=" generate a fasta file with synthetic "
                                                 " ribosome-protected fragments given a transcriptome")

    parser.add_argument("--footprint_length",
                        type=int,
                        default=29,
                        help="length of ribosome footprint (default: 29)")

    parser.add_argument("--output_fastq_prefix",
                        type=str,
                        default=None,
                        help="prefix of output fastq file (default: None)")

    parser.add_argument("gtf_file",
                        action="store",
                        help="gtf file containing the transcript models")

    parser.add_argument("fasta_file",
                        action="store",
                        help="fasta file containing the genome sequence")

    options = parser.parse_args()

    if options.output_fastq_prefix is None:
        options.output_fastq_prefix = options.gtf_file+'_%d.fq.gz'%options.footprint_length
    else:
        options.output_fastq_prefix = options.output_fastq_prefix+'_%d.fq.gz'%options.footprint_length

    return options

if __name__=="__main__":

    options = parse_args()

    qual = ''.join(['~' for r in xrange(options.footprint_length)])
    seq_handle = pysam.FastaFile(options.fasta_file)

    # load transcripts
    transcripts = load_data.load_gtf(options.gtf_file)
    tnames = transcripts.keys()

    fastq_handle = gzip.open(options.output_fastq_prefix, 'wb')
    for num,tname in enumerate(tnames):

        transcript = transcripts[tname]

        # get transcript DNA sequence
        sequence = seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

        # get forward strand reads
        if transcript.strand=="-":
            transcript.mask = transcript.mask[::-1]
            transcript.strand = "+"

        seq = ''.join(np.array(list(sequence))[transcript.mask].tolist())
        L = len(seq)
        positions = transcript.start + np.where(transcript.mask)[0]
        reads = [seq[i:i+options.footprint_length] 
                 for i in xrange(L-options.footprint_length+1)]
    
        # write synthetic reads
        fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n"%(transcript.chromosome, \
            position,transcript.strand,read,qual) \
            for position,read in zip(positions,reads)]))

        # get reverse strand reads
        transcript.mask = transcript.mask[::-1]
        transcript.strand = "-"
        seq = seq[::-1]
        seq = ''.join(utils.make_complement(seq))
        positions = transcript.start + transcript.mask.size - np.where(transcript.mask)[0]
        reads = [seq[i:i+options.footprint_length] 
                 for i in xrange(L-options.footprint_length+1)]

        # write synthetic reads
        fastq_handle.write(''.join(["@%s:%d:%s\n%s\n+\n%s\n"%(transcript.chromosome, \
            position,transcript.strand,read,qual) \
            for position,read in zip(positions,reads)]))

    seq_handle.close()
    fastq_handle.close()
