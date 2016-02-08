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

    parser.add_argument("--output_file",
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

    parser.add_argument("model_file",
                        action="store",
                        help="file name to the model parameters")

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

    if options.output_file is None:
        options.output_file = options.model_file+'bed12'

    return options

def write_inferred_cds(handle, transcript, state, frame, rna_sequence):

    posteriors = state.max_posterior*frame.posterior
    index = np.argmax(posteriors)
    tis = state.best_start[index]
    tts = state.best_stop[index]

    # output is not a valid CDS
    if tis is None or tts is None:
        return None

    posterior = int(posteriors[index]*10000) 
    protein = utils.translate(rna_sequence[tis:tts])
    # identify TIS and TTS in genomic coordinates
    if transcript.strand=='+':
        cdstart = transcript.start + np.where(transcript.mask)[0][tis]
        cdstop = transcript.start + np.where(transcript.mask)[0][tts]
    else:
        cdstart = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tts]
        cdstop = transcript.start + transcript.mask.size - np.where(transcript.mask)[0][tis]

    towrite = [transcript.chromosome, 
               transcript.start, 
               transcript.stop, 
               transcript.id, 
               posterior, 
               transcript.strand, 
               cdstart, 
               cdstop,
               protein, 
               len(transcript.exons), 
               ','.join(map(str,[e[1]-e[0] for e in transcript.exons]))+',', 
               ','.join(map(str,[transcript.start+e[0] for e in transcript.exons]))+',']
    handle.write(" ".join(map(str,towrite))+'\n')

    return None

def infer(options):

    # load the model
    handle = open(options.model_file, 'r')
    transition = cPickle.load(handle)
    emission = cPickle.load(handle)
    handle.close()

    # load transcripts
    transcript_models = load_data.load_gtf(options.gtf_file)
    transcript_names = transcript_models.keys()
    N = len(transcript_names)
    n = int(np.ceil(N/1000))
    
    # load data tracks
    genome_track = load_data.Genome(options.fasta_file, options.mappability_file)
    ribo_track = load_data.RiboSeq(options.riboseq_file)
    if options.rnaseq_file is not None:
        rnaseq_track = load_data.RnaSeq(options.rnaseq_file)

    # open output file handle
    # file in bed12 format
    handle = open(options.output_file,'w')
    towrite = ["chromosome", "start", "stop", "transcript_id", 
               "posterior", "strand", "cdstart", "cdstop", 
               "protein_seq", "num_exons", "exon_sizes", "exon_starts"]
    handle.write(" ".join(map(str,towrite))+'\n')

    for n in xrange(N):

        tnames = transcript_names[n*1000:(n+1)*1000]
        alltranscripts = [transcript_models[name] for name in tnames]

        # run inference on both strands independently

        # focus on positive strand
        for t in alltranscripts:
            if t.strand=='-':
                t.mask = t.mask[::-1]
                t.strand = '+'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]
        T = len(transcripts)
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            rna_sequences = genome_track.get_sequence(transcripts)
            for rna_sequence in rna_sequences:
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
            states, frames = ribohmm.infer_coding_sequence(footprint_counts, codon_flags, \
                                   rna_counts, rna_mappability, transition, emission)

            # write results
            ig = [write_inferred_cds(handle, transcript, state, frame, rna_sequence) \
                  for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences)]


        # focus on negative strand
        for t in alltranscripts:
            t.mask = t.mask[::-1]
            t.strand = '-'

        # check if all exons have at least 5 footprints
        exon_counts = ribo_track.get_exon_total_counts(alltranscripts)
        transcripts = [t for t,e in zip(alltranscripts,exon_counts) if np.all(e>=5)]
        T = len(transcripts)
        if T>0:

            # load sequence of transcripts and transform sequence data
            codon_flags = []
            rna_sequences = genome_track.get_sequence(transcripts)
            for rna_sequence in rna_sequences:
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
            states, frames = ribohmm.infer_coding_sequence(footprint_counts, codon_flags, \
                                   rna_counts, rna_mappability, transition, emission)

            # write results
            ig = [write_inferred_cds(handle, transcript, state, frame, rna_sequence) \
                  for transcript,state,frame,rna_sequence in zip(transcripts,states,frames,rna_sequences)]

    handle.close()
    ribo_track.close()
    if options.rnaseq_file is not None:
        rnaseq_track.close()
    genome_track.close()


if __name__=="__main__":

    options = parse_args()

    infer(options)
