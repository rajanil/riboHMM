import numpy as np
import pysam
import utils
import pdb

MIN_MAP_QUAL = 10

class Genome():

    def __init__(self, fasta_filename, map_filename):

        self._seq_handle = pysam.FastaFile(fasta_filename)
        self._map_handles = [pysam.TabixFile(map_filename+'_%d.gz'%r) 
                            for r in utils.READ_LENGTHS]

    def get_sequence(self, transcripts):

        sequences = []
        for transcript in transcripts:

            # get DNA sequence
            seq = self._seq_handle.fetch(transcript.chromosome, transcript.start, transcript.stop).upper()

            # get DNA sequence of transcript
            # reverse complement, if necessary
            if transcript.strand=="-":
                seq = seq[::-1]
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())
                seq = utils.make_complement(seq)
            else:
                seq = ''.join(np.array(list(seq))[transcript.mask].tolist())

            # get RNA sequence
            seq = ''.join(['U' if s=='T' else s for s in seq])
            sequences.append(seq)
            
        return sequences

    def get_mappability(self, transcripts):

        mappabilities = []
        for transcript in transcripts:

            # get mappable positions
            mappables = [np.zeros(transcript.mask.shape, dtype='bool') 
                         for r in utils.READ_LENGTHS]
            tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop)
                         for handle in self._map_handles]
            if transcript.strand=='+':
                offsets = [1,1,1,1]
            else:
                offsets = utils.READ_LENGTHS

            for tbx_iter,mappable,offset in zip(tbx_iters,mappables,offsets):

                for tbx in tbx_iter:

                    row = tbx.split('\t')
                    start = int(row[1]) - transcript.start + offset - 1
                    end = int(row[2]) - transcript.start + offset - 1
                    mappable[start:end] = True

            if transcript.strand=='+':
                mappables = np.array(mappables).T.astype('bool')
            else:
                mappables = np.array(mappables).T.astype('bool')[::-1]

            mappabilities.append(mappables[transcript.mask,:])

        return mappabilities
            
    def close(self):

        self._seq_handle.close()
        ig = [handle.close() for handle in self._map_handles]

class RiboSeq():

    def __init__(self, file_prefix):

        self._fwd_handles = [pysam.TabixFile(file_prefix+'_fwd.%d.gz'%r) 
                             for r in utils.READ_LENGTHS]
        self._rev_handles = [pysam.TabixFile(file_prefix+'_rev.%d.gz'%r) 
                             for r in utils.READ_LENGTHS]

    def get_counts(self, transcripts):

        read_counts = []
        for transcript in transcripts:

            rcounts = [np.zeros(transcript.mask.shape, dtype='int') for r in utils.READ_LENGTHS]
            if transcript.strand=='+':
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self._fwd_handles]
            else:
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self._rev_handles]

            for tbx_iter,counts in zip(tbx_iters,rcounts):

                for tbx in tbx_iter:

                    row = tbx.split('\t')
                    count = int(row[3])
                    asite = int(row[1]) - transcript.start
                    counts[asite] = count

            if transcript.strand=='+':
                rcounts = np.array(rcounts).T.astype(np.uint64)
            else:
                rcounts = np.array(rcounts).T.astype(np.uint64)[::-1]

            read_counts.append(rcounts[transcript.mask,:])

        return read_counts

    def get_total_counts(self, transcripts):

        read_counts = self.get_counts(transcripts)
        total_counts = np.array([counts.sum() for counts in read_counts])
        return total_counts

    def get_exon_total_counts(self, transcripts):

        exon_counts = []
        for transcript in transcripts:

            rcounts = [np.zeros(transcript.mask.shape, dtype='int') for r in utils.READ_LENGTHS]
            if transcript.strand=='+':
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self._fwd_handles]
            else:
                tbx_iters = [handle.fetch(transcript.chromosome, transcript.start, transcript.stop) \
                    for handle in self._rev_handles]

            for tbx_iter,counts in zip(tbx_iters,rcounts):

                for tbx in tbx_iter:

                    row = tbx.split('\t')
                    count = int(row[3])
                    asite = int(row[1]) - transcript.start
                    counts[asite] = count

            rcounts = np.array(rcounts).T.astype(np.uint64)
            exon_counts.append(np.array([rcounts[start:end,:].sum() 
                               for start,end in transcript.exons]))

        return exon_counts

    def close(self):

        ig = [handle.close() for handle in self._fwd_handles]
        ig = [handle.close() for handle in self._rev_handles]

class RnaSeq():

    def __init__(self, filename):

        self._handle = pysam.TabixFile(filename+'.gz')
        self.total = 0
        for chrom in self._handle.contigs:
            self.total += reduce(lambda x,y: x+y, (int(tbx.split('\t')[3]) for tbx in self._handle.fetch(chrom)))

    def get_total_counts(self, transcripts):

        total_counts = []
        for transcript in transcripts:

            tbx_iter = self._handle.fetch(transcript.chromosome, transcript.start, transcript.stop)
            if transcript.strand=='+':
                mask = transcript.mask
            else:
                mask = transcript.mask[::-1]

            counts = 0
            for tbx in tbx_iter:

                row = tbx.split('\t')
                site = int(row[1])-transcript.start
                count = int(row[3])
                if mask[site]:
                    counts += 1

            total_counts.append(max([1,counts])*1e6/float(transcript.L*self.total))

        total_counts = np.array(total_counts)
        return total_counts

    def close(self):

        self._handle.close()

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
            self.geneid = attr['gene_id']
        except KeyError:
            pass
        try:
            self.genename = attr['gene_name']
        except KeyError:
            pass
        try:
            self.ref_transcript_id = attr['reference_id']
        except KeyError:
            pass
        try:
            self.ref_gene_id = attr['ref_gene_id']
        except KeyError:
            pass
        try:
            self.genename = attr['ref_gene_name']
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
            self.start = min([self.start, self.exons[0][0]])
            self.stop = max([self.stop, self.exons[-1][-1]])

            # set transcript model
            self.exons = [(e[0]-self.start, e[1]-self.start) for e in self.exons]
            self.mask = np.zeros((self.stop-self.start,),dtype='bool')
            ig = [self.mask.__setslice__(start,stop,True) for (start,stop) in self.exons]
            if self.strand=='-':
                self.mask = self.mask[::-1]

            self.L = self.mask.sum()

        else:

            # no exons for transcript; remove
            raise ValueError

def load_gtf(filename):

    transcripts = dict()
    handle = open(filename, "r")

    for line in handle:
        # remove comments
        if line.startswith('#'):
            continue

        # read data
        data = line.strip().split("\t")
        attr = dict([(ln.split()[0],eval(ln.split()[1])) for ln in data[8].split(';')[:-1]])

        # identify chromosome of the transcript
        if data[0].startswith('c'):
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

            if data[2]=='transcript':
                # initialize new transcript
                transcripts[transcript_id] = Transcript(data, attr)
                
    handle.close()

    # generate transcript models
    keys = transcripts.keys()
    for key in keys:
        try:
            transcripts[key].generate_transcript_model()
        except ValueError:
            del transcripts[key]

    return transcripts

        
