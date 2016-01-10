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

    def __init__(self, filename):

        self.handle = pysam.Samfile(filename, "rb")

    def get_counts(self, transcripts):

        read_counts = []
        for transcript in transcripts:

            counts = dict([(r,np.zeros(transcript.mask.shape, dtype='int')) for r in utils.READ_LENGTHS])
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

class Gene():

    def __init__(self, line, attr):
        try:
            self.name = attr['gene_name']
        except KeyError:
            # not a field in some gtf files
            pass
        self.chromosome = line[0]
        self.transcripts = dict()
        if line[6] in ['+','-']:
            self.strand = line[6]
        else:
            self.strand = '+'
        self.start = 0
        self.stop = 0
        try:
            self.type = attr['gene_type']
        except KeyError:
            pass
        try:
            self.type = attr['gene_biotype']
        except KeyError:
            pass
        self.gene_set = False

    def add_gene_info(self, line):
        self.start = int(line[3])-1-exten
        self.stop = int(line[4])+exten
        self.gene_set = True

    def add_transcript(self, line, attr):
        transcript_id = attr['transcript_id']
        self.transcripts[transcript_id] = Transcript(line, attr)
        self.transcripts[transcript_id].geneid = attr['gene_id']
        try:
            self.transcripts[transcript_id].genename = attr['gene_name']
        except KeyError:
            # some gtf files don't have this field
            pass

    def add_exon(self, line, attr):
        transcript_id = attr['transcript_id']
        self.transcripts[transcript_id].add_exon(line)
        try:
            self.transcripts[transcript_id].exonrpkms.append(attr['cov'])
        except KeyError:
            pass

    def add_start_codon(self, line, attr):
        transcript_id = attr['transcript_id']
        self.transcripts[transcript_id].add_start_codon(line)

    def add_stop_codon(self, line, attr):
        transcript_id = attr['transcript_id']
        self.transcripts[transcript_id].add_stop_codon(line)

    def generate_gene_models(self):
        ig = [self.transcripts[key].generate_transcript_model() for key in self.transcripts.keys()]
        newstart = np.min([t.start for t in self.transcripts.values() if t.transcript_set])
        newstop = np.max([t.stop for t in self.transcripts.values() if t.transcript_set])
        if not self.gene_set:
            self.start = newstart
            self.stop = newstop
            self.gene_set = True
        if newstart<self.start or newstop>self.stop:
            self.gene_set = False
            utils.pdb.set_trace()

    def __len__(self):

        return self.stop-self.start

class Transcript():

    def __init__(self, line, attr):

        self.id = attr['transcript_id']
        self.chromosome = line[0]
        self.transcript_set = False
        if line[6] in ['+','-']:
            self.strand = line[6]
        else:
            self.strand = '+'
        self.start = None
        self.stop = None
        self.cdstart = None
        self.cdstop = None
        try:
            self.type = attr['transcript_type']
        except KeyError:
            pass
        try:
            self.type = attr['gene_biotype']
        except KeyError:
            pass
        self.exons = []
        self.exonrpkms = []
        self.has_CDS = False
        self.genename = ''
        self.geneid = ''
        self.proteinid = ''
        self.ref_transcript_id = None
        self.ref_gene_id = None
        self.fpkm = 0
        self.cov = 0

    def add_transcript_info(self, line, attr):

        self.start = int(line[3])-1-exten
        self.stop = int(line[4])+exten
        self.transcript_set = True

    def add_exon(self, line):
        exon = (int(line[3])-1, int(line[4]))
        self.exons.append(exon)

    def add_start_codon(self, line):
        if line[6]=='+':
            self.cdstart = int(line[3])-1
        else:
            self.cdstop = int(line[4])

    def add_stop_codon(self, line):
        if line[6]=='+':
            self.cdstop = int(line[4])
        else:
            self.cdstart = int(line[3])-1

    def generate_transcript_model(self):

        if len(self.exons)>0:

            # order exons
            order = np.argsort(np.array([e[0] for e in self.exons]))
            self.order = order
            self.exons = [[self.exons[o][0],self.exons[o][1]] for o in order]

            start = self.exons[0][0]-exten
            stop = self.exons[-1][-1]+exten
            if not self.transcript_set:
                self.start = start
                self.stop = stop
                self.transcript_set = True
            if self.start>start or self.stop<stop:
                self.transcript_set = False
                utils.pdb.set_trace()

            self.exons = [(e[0]-self.start-exten, e[1]-self.start-exten) for e in self.exons]

            self.mask = np.zeros((self.stop-self.start,),dtype='bool')
            ig = [self.mask.__setslice__(start+exten,stop+exten,True) for (start,stop) in self.exons]
            #self.mask[:exten] = True
            #self.mask[-exten:] = True
            if self.strand=='-':
                self.mask = self.mask[::-1]
            mindex = np.where(self.mask)[0]
            self.boundaries = np.where((mindex[:-1]-mindex[1:])<-1)[0]+1

            if self.cdstart is not None and self.cdstop is not None:
                self.has_CDS = True
                if self.strand=='+':
                    self.utr5 = self.mask[:self.cdstart-self.start].sum()
                    self.utr3 = self.mask[:(self.cdstop-self.start)].sum()
                else:
                    self.utr5 = self.mask[:self.stop-self.cdstop].sum()
                    self.utr3 = self.mask[:(self.stop-self.cdstart)].sum()
            else:
                self.utr5 = None
                self.utr3 = None
            self._utr5 = self.utr5
            self._utr3 = self.utr3

    def modify_CDS(self, start, stop):

        self._utr5 = self.utr5
        self._utr3 = self.utr3
        self.utr5 = start
        self.utr3 = stop

    def revert_CDS(self):

        self.utr5 = self._utr5
        self.utr3 = self._utr3

    def __len__(self):

        return self.utr3-self.utr5

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

        # is the transcript in the gene dictionary?
        transcript_id = attr['transcript_id']
        try:
            transcripts[transcript_id]
            if data[2]=='exon':
                transcripts[transcript_id].add_exon(data, attr)
            else:
                print "Unknown annotation"
                utils.pdb.set_trace()
        except KeyError:
            genes[geneid].add_transcript(data, attr)
            if data[2]=='transcript':
                genes[geneid].transcripts[transcript_id].add_transcript_info(data, attr)
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

            else:
                utils.pdb.set_trace()
    handle.close()

    # generate transcript models;
    # set gene and transcript attributes, if not set already.
    for key,value in genes.iteritems():
        genes[key].generate_gene_models()

    genedict = dict([(gene,[val.transcripts[key] for key in np.sort(val.transcripts.keys()).tolist() \
        if val.transcripts[key].transcript_set]) for gene,val in genes.iteritems()])

    return transcripts

        
