import argparse
import cPickle
import warnings
import pdb

import numpy as np

import load_data
import ribohmm
import seq
import utils

# ignore warnings with these expressions
warnings.filterwarnings('ignore', '.*overflow encountered.*',)
warnings.filterwarnings('ignore', '.*divide by zero.*',)
warnings.filterwarnings('ignore', '.*invalid value.*',)

def parse_args():
    parser = argparse.ArgumentParser(description=" infers the translated sequences "
                                     " from ribosome profiling data and RNA sequence data; "
                                    " RNA-seq data can also be used if available ")

    parser.add_argument("--model_file",
                        type=str,
                        default=None,
                        help="output file name to store the model parameters")

    parser.add_argument("--rnaseq_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with counts of RNA-seq reads")

    parser.add_argument("--mappability_file",
                        type=str,
                        default=None,
                        help="prefix of tabix file with mappability information")

    parser.add_argument("fasta_file",
                        action="store",
                        help="fasta file containing the genome sequence")

    parser.add_argument("gtf_file",
                        action="store",
                        help="gtf file containing the assembled transcript models")

    parser.add_argument("riboseq_file",
                        action="store",
                        help="prefix of tabix files with counts of ribosome footprints")

    options = parser.parse_args()

    return options


def infer(options):

    # load the model
    handle = open(options.model_file, 'r')
    transition = cPickle.load(handle)
    emission = cPickle.load(handle)
    handle.close()

    # load transcripts
    transcript_models = load_data.load_gtf(options.gtf_file)

    # load data tracks
    genome_track = load_data.Genome(options.fasta_file, options.mappability_file)
    ribo_track = load_data.RiboSeq(options.riboseq_file)
    if options.rnaseq_file is not None:
        rnaseq_track = load_data.RnaSeq(options.rnaseq_file)

    states = dict()
    frames = dict()

    genenames = transcript_models.keys()
    for genename in genenames:

        # run inference on both strands independently

        # focus on positive strand
        for t in transcript_models[gene]:
            t.strand = '+'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(transcript_models[gene])
        transcripts = [t for t,e in zip(transcript_models[gene],exon_counts) if e>=5]
        T = len(transcripts)
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            for rna_sequence in genome_track.get_sequence(transcripts):
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            if options.rnaseq_file is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            if options.mappability_file is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            # run the learning algorithm
            fwdstate, fwdframe = ribohmm.infer_coding_sequence(footprint_counts, codon_flags, \
                                   rna_counts, rna_mappability, transition, emission)

        # focus on negative strand
        for t in transcript_models[gene]:
            t.strand = '-'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(transcript_models[gene])
        transcripts = [t for t,e in zip(transcript_models[gene],exon_counts) if e>=5]
        T = len(transcripts)
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            for rna_sequence in genome_track.get_sequence(transcripts):
                sequence = seq.RnaSequence(rna_sequence)
                codon_flags.append(sequence.mark_codons())

            # load footprint count data in transcripts
            footprint_counts = ribo_track.get_counts(transcripts)

            # load transcript-level rnaseq RPKM
            if options.rnaseq_file is None:
                rna_counts = np.ones((T,), dtype='float')
            else:
                rna_counts = rnaseq_track.get_total_counts(transcripts)

            # load mappability of transcripts; transform mappability to missingness
            if options.mappability_file is not None:
                rna_mappability = genome_track.get_mappability(transcripts)
            else:
                rna_mappability = [np.ones(c.shape,dtype='bool') for c in footprint_counts]

            # run the learning algorithm
            revstate, revframe = ribohmm.infer_coding_sequence(footprint_counts, codon_flags, \
                                   rna_counts, rna_mappability, transition, emission)

    ribo_track.close()
    if options.rnaseq_file is not None:
        rnaseq_track.close()
    genome_track.close()


if __name__=="__main__":

    options = parse_args()

    infer(options)
