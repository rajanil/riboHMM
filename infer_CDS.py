import numpy as np
import scipy.stats as stats
import argparse
import cPickle
import loadutils
import cribohmmmod as ribohmm
import utils
import time

def parse_args():
    parser = argparse.ArgumentParser(description="decodes the HMM"
                                     "to infer the coding sequence"
                                     "from ribosome profiling data")

    parser.add_argument("--stop", default='firststop',
                        help="specify whether to allow stop codon readthrough {firststop / readthrough}")

    parser.add_argument("--assembly", default='denovo',
                        help="specify the species/assembly to run the model on")

    parser.add_argument("--bound", type=int,
                        help="specify set of genes to run")

    parser.add_argument("--set", type=int, default=0,
                        help="specify learning set")

    parser.add_argument("--subsample", default=None,
                        help="binomially subsample the ribosome footprint data by this fraction")

    options = parser.parse_args()

    return options


def infer(options):

    # load the model
    filename = "%s/cache/ribohmmmod_model_%s_%s_gene_%s.pkl"%(utils.PATH,options.assembly,options.stop,options.subsample)
    handle = open(filename, 'r')
    transition = cPickle.load(handle)
    emission = cPickle.load(handle)
    handle.close()

    # load transcripts
    handle = open("%s/cache/annotation_%s_%d.pkl"%(utils.PATH,options.assembly,options.bound),'r')
    annotation = cPickle.load(handle)
    handle.close()
    genes = annotation.keys()

    ribo_tracks = [loadutils.RiboSeq(sample='combined_%d'%i) for i in utils.READ_LENGTHS]
    ribototal = np.sum([track.total for track in ribo_tracks])
    ribo_totaltrack = loadutils.RiboSeq(sample='combined_28_31')
    seq_track = loadutils.Sequence()
    rnaseq_track = loadutils.RnaSeq(sample="riboseq/RNAseqGeuvadis_combined_read_start_count")

    kozak_data = np.load("%s/cache/kozak_sequence.npz"%utils.PATH)
    freq = dict([(char,row) for char,row in zip(['A','U','G','C'],kozak_data['freq'])])
    altfreq = dict([(char,row) for char,row in zip(['A','U','G','C'],kozak_data['altfreq'])])
    kozak_data.close()
    for char in ['A','U','G','C']:
        freq[char][9:12] = altfreq[char][9:12]

    states = dict()
    frames = dict()
    starttime = time.time()
    for g,gene in enumerate(genes):

        transcripts = [t for t in annotation[gene]]

        # focus on positive strand
        for t in transcripts:
            if t.strand=='-':
                t.mask = t.mask[::-1]
                t.strand = '+'
            else:
                t.strand = '+'

        if len(transcripts)>0:
        
            # load read and sequence data
            readdata = [ribo_track.get_read(transcripts) for ribo_track in ribo_tracks]
            data = [np.array(d).T.astype(np.uint64) for d in zip(*readdata)]

            if options.subsample is not None:
                fraction = int(options.subsample)/580.
                newdata = []
                for datum in data:
                    newdatum = np.zeros(datum.shape, dtype='int')
                    newdatum[datum>0] = stats.binom.rvs(datum[datum>0].astype('int'), fraction)
                    newdata.append(newdatum.astype(np.uint64))
                data = newdata

            # load scale
            rnareads = rnaseq_track.get_transcript_reads(transcripts)

            # specify transition probabilities
            sequences = seq_track.get_sequence(transcripts)
            codon_id = [dict([('kozak',utils.start_kozak(sequence, freq, altfreq)), \
                ('start',utils.start_triplet_transform(sequence)), \
                ('stop',utils.stop_triplet_transform(sequence))]) \
                for sequence in sequences]

            mappability = seq_track.get_mappability(transcripts)
            # transform mappability to missingness
            missing = [np.logical_not(mappable).T for mappable in mappability]
            del mappability

            # run inference
            fwdstate, fwdframe = ribohmm.infer_coding_sequence(data, codon_id, rnareads, missing, transition, emission)

            # retain inference for transcripts that have atleast 2 footprints in each exon
            exon_totals = ribo_totaltrack.get_exon_totals(transcripts)
            for state,frame,total in zip(fwdstate, fwdframe, exon_totals):
                if not np.all(total>5):
                    state.best_start = [None, None, None]
                    state.best_stop = [None, None, None]

        else:

            fwdstate = []
            fwdframe = []

        print gene, [((state.best_start[frame.posterior.argmax()], state.best_stop[frame.posterior.argmax()]), \
                     state.max_posterior[frame.posterior.argmax()]*frame.posterior.max(), '+', t.ref_transcript_id) \
                    for state,frame,t in zip(fwdstate,fwdframe,transcripts)]

        # focus on negative strand
        transcripts = [t for t in annotation[gene]]
        for t in transcripts:
            t.mask = t.mask[::-1]
            t.strand = '-'

        if len(transcripts)>0:

            # load read and sequence data
            readdata = [ribo_track.get_read(transcripts) for ribo_track in ribo_tracks]
            data = [np.array(d).T.astype(np.uint64) for d in zip(*readdata)]

            if options.subsample is not None:
                fraction = int(options.subsample)/580.
                newdata = []
                for datum in data:
                    newdatum = np.zeros(datum.shape, dtype='int')
                    newdatum[datum>0] = stats.binom.rvs(datum[datum>0].astype('int'), fraction)
                    newdata.append(newdatum.astype(np.uint64))
                data = newdata

            # load scale
            rnareads = rnaseq_track.get_transcript_reads(transcripts)

            # specify transition probabilities
            sequences = seq_track.get_sequence(transcripts)
            codon_id = [dict([('kozak',utils.start_kozak(sequence, freq, altfreq)), \
                ('start',utils.start_triplet_transform(sequence)), \
                ('stop',utils.stop_triplet_transform(sequence))]) \
                for sequence in sequences]

            mappability = seq_track.get_mappability(transcripts)
            # transform mappability to missingness
            missing = [np.logical_not(mappable).T for mappable in mappability]
            del mappability

            # run inference
            revstate, revframe = ribohmm.infer_coding_sequence(data, codon_id, rnareads, missing, transition, emission)

            # retain inference for transcripts that have atleast 2 footprints in each exon
            exon_totals = ribo_totaltrack.get_exon_totals(transcripts)
            for state,frame,total in zip(revstate, revframe, exon_totals):
                if not np.all(total>5):
                    state.best_start = [None, None, None]
                    state.best_stop = [None, None, None]

        else:
            revstate = []
            revframe = []

        print gene, [((state.best_start[frame.posterior.argmax()], state.best_stop[frame.posterior.argmax()]), \
                     state.max_posterior[frame.posterior.argmax()]*frame.posterior.max(), '-', t.ref_transcript_id) \
                    for state,frame,t in zip(revstate,revframe,transcripts)]

        states[gene] = [(sf,sr) for sf,sr in zip(fwdstate,revstate)]
        frames[gene] = [(ff,fr) for ff,fr in zip(fwdframe,revframe)]
        if (g+1)%100==0:
            print time.time()-starttime
            starttime = time.time()

    rnaseq_track.close()
    seq_track.close()
    ig = [ribo_track.close() for ribo_track in ribo_tracks]
    ribo_totaltrack.close()

    # save inference
    filename = "%s/cache/ribohmmmod_inference_mORF_%s_%s_%d_gene_%s.pkl"%(utils.PATH,options.assembly,options.stop,options.bound,options.subsample)
    handle = open(filename,'w')
    cPickle.Pickler(handle,protocol=2).dump(states)
    cPickle.Pickler(handle,protocol=2).dump(frames)
    handle.close()


if __name__=="__main__":

    options = parse_args()

    infer(options)
