import numpy as np
import load_data
import pysam
import ribohmm
import argparse
import utils


def parse_args():
    parser = argparse.ArgumentParser(description=" learns the parameters of riboHMM to infer translation "
                                     " from ribosome profiling data and RNA sequence data; "
                                    " RNA-seq data can also be used if available ")

    parser.add_argument("--restarts",
                        type=int,
                        default=1,
                        help="number of re-runs of the algorithm (default: 1)")

    parser.add_argument("--mintol",
                        type=float,
                        default=1e-6,
                        help="convergence criterion for change in per-base marginal likelihood (default: 1e-4)")

    parser.add_argument("--batch",
                        type=int,
                        default=1000,
                        help="number of transcripts used for learning model parameters.")

    parser.add_argument("--model_output_file",
                        type=str,
                        default=None,
                        help="file name to store the model parameters")

    parser.add_argument("--log_file",
                        type=str,
                        default=None,
                        help="file name to store some statistics of the EM algorithm ")

    parser.add_argument("--rnaseq_bam_file",
                        type=str,
                        default=None,
                        help="bam file from RNA-seq assay")

    parser.add_argument("--mappability_file",
                        type=str,
                        default=None,
                        help="file with mappability information")

    parser.add_argument("genome_fasta",
                        action="store",
                        help="fasta file containing the genome sequence")

    parser.add_argument("gtf_input_file",
                        action="store",  
                        help="gtf file containing the assembled transcript models")

    parser.add_argument("riboseq_bam_file",
                        action="store",
                        help="bam file from ribosome profiling assay.")

    options = parser.parse_args()

    return options

def select_transcripts(options):
    raise NotImplementedError
    
    # load all transcripts
    transcript_models = load_data.load_transcripts(options.gtf_input_file)

    # get translation level in all transcripts
    transcript_counts = []
    bamhandle =
    for transcript in transcript_models:


    bamhandle.close()

    # select top transcripts
    transcripts = []
    transcript_bounds = dict()
    order = np.argsort(transcript_counts)[::-1]
    for index in order:
        transcript = transcript_models[index]

        # test if transcript overlaps any previous transcript
        overlap = False
        try:
            for bound in transcript_bounds[transcript.chromosome]:
                if not (transcript.stop<bound[0] or transcript.start>bound[1]):
                    overlap = True
                    break
        except KeyError:
            pass

        if not overlap:
            transcripts.append(transcript)
            transcript_bounds[transcript.chromosome].append([transcript.start, transcript.stop])
            if len(transcripts)>=options.batch:
                break

    return transcripts

def learn(options):

    # select transcripts for learning parameters
    transcripts = select_transcripts(options)

    # load footprint count data in transcripts
    ribo_track = load_data.RiboSeq(options.riboseq_bam_file)
    countdata = ribo_track.get_counts(transcripts)
    footprint_counts = [np.array(d).T.astype(np.uint64) for d in zip(*readdata)]
    ribo_track.close()

    # load rnaseq count data in transcripts
    if options.rnaseq_bam_file is not None:
        rnaseq_track = load_data.RnaSeq(options.rnaseq_bam_file)
        rna_counts = rnaseq_track.get_counts(transcripts)
        rnaseq_track.close()
    else:
        rna_counts = None

    # load pre-computed Kozak model
    kozak_model = np.load("%s/cache/kozak_sequence.npz"%utils.PATH)
    freq = dict([(char,row) for char,row in zip(['A','U','G','C'],kozak_model['freq'])])
    altfreq = dict([(char,row) for char,row in zip(['A','U','G','C'],kozak_model['altfreq'])])
    kozak_data.close()
    for char in ['A','U','G','C']:
        freq[char][9:12] = altfreq[char][9:12]

    # load sequence of transcripts and transform sequence data
    genome_track = load_data.Genome(options.genome_fasta, options.mappability_file)
    rna_sequences = genome_track.get_sequence(transcripts)
    codon_id = [dict([('kozak',utils.compute_kozak_scores(sequence, freq, altfreq)), \
        ('start',utils.mark_start_codons(sequence)), \
        ('stop',utils.mark_stop_codons(sequence))]) \
        for sequence in rna_sequences]
    total_bases = np.sum([len(seq) for seq in rna_sequences])
    del sequences

    # load mappability of transcripts; transform mappability to missingness
    if options.mappability_file is not None:
        rna_mappability = genome_track.get_mappability(transcripts)
        missing = [np.logical_not(mappable).T for mappable in rna_mappability]
        del rna_mappability
    else:
        missing = None
    genome_track.close()

    # run the learning algorithm
    print "Learning using %d transcripts, %d footprints covering %d bases"%(len(transcripts),total_footprints,total_bases)
    transition, emission = ribohmm.learn_parameters(footprint_counts, codon_id, \
        rna_counts, missing, options.restarts, options.mintol)

    # output model parameters


if __name__=="__main__":

    options = parse_args()

    learn(options)
