import numpy as np
import pysam
import utils

MIN_MAP_QUAL = 10

class Genome():

    def __init__(self, fasta_filename, map_filename):

        self._seq_handle = pysam.FastaFile(fasta_filename)

    def get_sequence(self, transcripts):

        sequences = []
        for transcript in transcripts:

            # get DNA sequence
            seq = self._seq_handle.fetch(transcript.chromsome, transcript.start, transcript.stop)

            # get DNA sequence of transcript
            # reverse complement, if necessary
            if transcript.strand=="-":
                seq = seq[::-1]
                seq = seq[transcript.mask]
                seq = utils.make_complement(seq)
            else:
                seq = seq[transcript.mask]

            # get RNA sequence
            seq = ''.join(['U' if s=='T' else s for s in seq])
            sequences.append(seq)
            
        return sequences

    def get_mappability(self, transcripts):

        raise NotImplementedError

    def close(self):

        self._seq_handle.close()
        #self._map_handle.close()

class RiboSeq():

    def __init__(self, file_prefix):

        self.fwd_handles = [pysam.TabixFile(file_prefix+'_fwd.%d.bgzf'%r) for r in utils.READ_LENGTHS]
        self.rev_handles = [pysam.TabixFile(file_prefix+'_rev.%d.bgzf'%r) for r in utils.READ_LENGTHS]

    def get_counts(self, transcripts):

        read_counts = []
        for transcript in transcripts:

            counts = []
            if transcript.strand=='+':
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self.fwd_handles]
            else:
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self.rev_handles]

            for tbx_iter in tbx_iters:
                for tbx in tbx_iter:

                if transcript.strand=='+':
                    # skip reverse strand reads
                    if read.is_reverse:
                        continue
                    else:
                        asite = read.positions[11]-transcript.start
                else:
                    # skip forward strand reads
                    if not read.is_reverse:
                        continue
                    else:
                        asite = transcript.stop-read.positions[-11]

                # check if A-site is within transcript 
                if asite>=0:
                    # only keep footprints of specific lengths
                    try:
                        counts[read.rlen][asite] += 1
                    except KeyError:
                        pass

            read_counts.append(np.array([counts[r][transcript.mask] \
                for r in utils.READ_LENGHTS]).T.astype(np.uint64))

        return read_counts

    def get_total_counts(self, transcripts):

        read_counts = self.get_counts(transcripts)
        total_counts = np.array([counts.sum() for counts in read_counts])
        return total_counts

    def get_exon_total_counts(self, transcripts):

        exon_counts = []
        for transcript in transcripts:

            counts = np.zeros(transcript.mask.shape, dtype='int')
            sam_iter = self._handle.fetch(reference=transcript.chromosome, start=transcript.start, end=transcript.stop)

            for read in sam_iter:

                # skip read if unmapped
                if read.is_unmapped:
                    continue

                # skip read, if mapping quality is low
                if read.mapq < MIN_MAP_QUAL:
                    continue

                if transcript.strand=='+':
                    # skip reverse strand reads
                    if read.is_reverse:
                        continue
                    else:
                        asite = read.positions[11]-transcript.start
                else:
                    # skip forward strand reads
                    if not read.is_reverse:
                        continue
                    else:
                        asite = read.positions[-11]-transcript.start

                # check if A-site is within transcript 
                if asite>=0:
                    counts[asite] += 1

            exon_counts.append(np.array([counts[start:end].sum() for start,end in transcript.exons])

        return exon_counts

    def close(self):

        self.handle.close()

class RnaSeq():

    def __init__(self, filename):

        self.handle = pysam.Samfile(filename, "rb")

    def get_total_counts(self, transcripts):

        total_counts = []
        for transcript in transcripts:

            counts = 0
            sam_iter = self._handle.fetch(reference=transcript.chromosome, start=transcript.start, end=transcript.stop)
            if transcript.strand=='+':
                mask = transcript.mask
            else:
                mask = transcript.mask[::-1]

            for read in sam_iter:

                # skip read if unmapped
                if read.is_unmapped:
                    continue

                # skip read, if mapping quality is low
                if read.mapq < MIN_MAP_QUAL:
                    continue

                if read.is_reverse:
                    site = read.pos - transcript.start
                else:
                    site = read.pos + read.alen - 1 - transcript.start
                
                if mask[site]:
                    count += 1

            total_counts.append(max([1,count])*1e6/transcript.L/self.total)

        total_counts = np.array(total_counts)
        return total_counts

    def close(self):

        self._track.close()

class Transcript():

    def __init__(self, line, attr):

        self.id = attr['transcript_id']
        self.chromosome = line[0]
        self.start = int(line[3])-1
        self.stop = int(line[4])

        # if not specified, transcript is on positive strand
        if line[6] in ['+','-']:
            self.strand = line[6]
        else:
            self.strand = '+'

        self.cdstart = None
        self.cdstop = None
        self.exons = []
        self.has_CDS = False
        self.proteinid = ''

        # add attribute fields that are available
        try:
            self.type = attr['transcript_type']
        except KeyError:
            pass
        try:
            self.type = attr['gene_biotype']
        except KeyError:
            pass
        try:
            self.transcripts[transcript_id].geneid = attr['gene_id']
        except KeyError:
            pass
        try:
            self.transcripts[transcript_id].genename = attr['gene_name']
        except KeyError:
            pass
        try:
            genes[geneid].transcripts[transcript_id].ref_transcript_id = attr['reference_id']
        except KeyError:
            pass
        try:
            genes[geneid].transcripts[transcript_id].ref_gene_id = attr['ref_gene_id']
        except KeyError:
            pass
        try:
            genes[geneid].transcripts[transcript_id].genename = attr['ref_gene_name']
        except KeyError:
            pass

    def add_exon(self, line):
        exon = (int(line[3])-1, int(line[4]))
        self.exons.append(exon)

    def generate_transcript_model(self):

        if len(self.exons)>0:

            # order exons
            order = np.argsort(np.array([e[0] for e in self.exons]))
            self.exons = [[self.exons[o][0],self.exons[o][1]] for o in order]

            # extend transcript boundaries, if needed
            transcript.start = min([transcript.start, self.exons[0][0]])
            transcript.stop = max([transcript.stop, self.exons[-1][-1]])

            # set transcript model
            self.exons = [(e[0]-self.start, e[1]-self.start) for e in self.exons]
            self.mask = np.zeros((self.stop-self.start,),dtype='bool')
            ig = [self.mask.__setslice__(start,stop,True) for (start,stop) in self.exons]
            if self.strand=='-':
                self.mask = self.mask[::-1]

        else:

            # no exons for transcript; remove
            raise ValueError

    def __len__(self):

        return self.mask.sum()

def load_gtf(filename):

    transcripts = dict()
    handle = open(filename, "r")

    for line in handle:
        # remove comments
        if '#'==line[0]:
            continue

        # read data
        data = line.strip().split("\t")
        attr = dict([(ln.split()[0],eval(ln.split()[1])) for ln in data[8].split(';')[:-1]])

        # identify chromosome of the transcript
        if data[0][0]=='c':
            chrom = data[0]
        else:
            chrom = 'chr%s'%data[0]
        data[0] = chrom

        transcript_id = attr['transcript_id']
        try:
        
            # if transcript is in dictionary, only parse exons
            transcripts[transcript_id]
            if data[2]=='exon':
                transcripts[transcript_id].add_exon(data)
            else:
                pass
        
        except KeyError:

            if not data[2]=='transcript':
                print "unknown annotation; poor ordering"
                pdb.set_trace()

            # initialize new transcript
            transcripts[transcript_id] = Transcript(line, attr)
                
    handle.close()

    # generate transcript models
    keys = transcripts.keys()
    for key in keys:
        try:
            transcripts[key].generate_transcript_model()
        except ValueError:
            del transcripts[key]

    return transcripts

        
