import numpy as np
import utils

class Genome():

    def __init__(self, fasta_file, map_file):

        self._seq_track = gdb.open_track("seq")
        self._map_tracks = dict()
        for rlen in utils.READ_LENGTHS:
            self._map_tracks[rlen] = gdb.open_track("riboseq/mappability_%dbp"%rlen)
        self._cons_track = gdb.open_track("phastcons/placental_46way")

        self.genome = dict()
        self.mappability = dict([(rlen,dict()) for rlen in utils.READ_LENGTHS])
        self.conservation = dict()
        for chromosome in chromosomes:
            self.genome[chromosome.name] = self._seq_track.get_array(chromosome.name)
            for rlen,track in self._map_tracks.iteritems():
                self.mappability[rlen][chromosome.name] = track.get_array(chromosome.name)
            try:
                self.conservation[chromosome.name] = self._cons_track.get_array(chromosome.name)
            except AttributeError:
                pass

    def get_sequence(self, transcripts, CDSstart=False, CDSstop=False, short=4, long=30):

        if CDSstart:
            offset = 0 #12
            left = short
            right = long
        elif CDSstop:
            offset = 0 #15
            right = short
            left = long

        seqs = []
        for transcript in transcripts:
            seq = self.genome[transcript.chromosome][transcript.start:transcript.stop]
            if transcript.strand=="-":
                seq = seq[::-1]
            seq = seq[transcript.mask]

            L = seq.size
            if CDSstart:
                center = transcript.utr5
                seq = seq[max([0,center-offset-left]):min([center-offset+right,L])]
                if seq.size!=(left+right):
                    seq = np.hstack((78*np.ones((left+right-seq.size,), dtype=np.uint16),seq))

            elif CDSstop:
                center = transcript.utr3
                seq = seq[max([0,center-offset-left]):min([center-offset+right,L])]
                if seq.size!=(left+right):
                    seq = np.hstack((seq,78*np.ones((left+right-seq.size,),dtype=np.uint16)))

            if transcript.strand=='-':
                seq = utils.make_complement(seq)

            seq = ['U' if utils.DNA_MAP[s]=='T' else utils.DNA_MAP[s] for s in seq]
            seqs.append(''.join(seq))
            
        return seqs

    def get_mappability(self, transcripts, CDSstart=False, CDSstop=False, short=4, long=30):

        if CDSstart:
            offset = 0 #12
            left = short
            right = long
        elif CDSstop:
            offset = 0 #15
            right = short
            left = long

        mapps = []
        for transcript in transcripts:
            if transcript.strand=='+':
                mapp = np.array([self.mappability[rlen][transcript.chromosome][transcript.start-12:transcript.stop-12][transcript.mask] for rlen in utils.READ_LENGTHS])
            else:
                mapp = np.array([self.mappability[rlen][transcript.chromosome][transcript.start-rlen+1+12:transcript.stop-rlen+1+12][::-1][transcript.mask] for rlen in utils.READ_LENGTHS])

            L = mapp.size
            if CDSstart: 
                center = transcript.utr5
                mapp = mapp[:,max([0,center-offset-left]):min([center-offset+right,L])]
                if mapp.shape[1]!=(left+right):
                    mapp = np.hstack((-1*np.ones((len(utils.READ_LENGTHS),left+right-mapp.size), dtype=np.uint16),mapp))

            elif CDSstop:
                center = transcript.utr3
                mapp = mapp[:,max([0,center-offset-left]):min([center-offset+right,L])]
                if mapp.shape[1]!=(left+right):
                    mapp = np.hstack((mapp,-1*np.ones((len(utils.READ_LENGTHS),left+right-mapp.size),dtype=np.uint16)))

            mapps.append(mapp)
            
        return mapps

    def close(self):

        self._seq_track.close()
        for key,track in self._map_tracks.iteritems():
            track.close()
        self._cons_track.close()

class RiboSeq():

    def __init__(self, sample=None, treated=False):

        if treated:
            self._fwd_track = gdb.open_track("riboseq/%s_treated_fwd"%sample)
            self._rev_track = gdb.open_track("riboseq/%s_treated_rev"%sample)
        else:
            self._fwd_track = gdb.open_track("riboseq/%s_untreated_fwd"%sample)
            self._rev_track = gdb.open_track("riboseq/%s_untreated_rev"%sample)

        self.forward = dict()
        self.reverse = dict()

        for chromosome in chromosomes:
            self.forward[chromosome.name] = self._fwd_track.get_array(chromosome.name)
            self.reverse[chromosome.name] = self._rev_track.get_array(chromosome.name)
        self.total = gdb.get_track_stat(self._fwd_track).sum + gdb.get_track_stat(self._rev_track).sum

    def get_read(self, transcripts, CDSstart=False, CDSstop=False, CDScenter=False, short=51, long=54):

        if CDSstart:
            left = short
            right = long
        elif CDSstop:
            right = short
            left = long
        elif CDScenter:
            left = short
            right = long

        reads = []
        for transcript in transcripts:
            if transcript.strand=='+':
                tread = self.forward[transcript.chromosome][transcript.start-12:transcript.stop-12][transcript.mask] #-12
            else:
                tread = self.reverse[transcript.chromosome][transcript.start+12:transcript.stop+12][::-1][transcript.mask] #+12

            L = tread.size
            if CDSstart:
                center = transcript.utr5
                read = tread[max([0,center-left]):min([center+right,L])]
                if read.size!=(left+right):
                    read = np.hstack((-1*np.ones((left+right-read.size,),dtype=np.uint16),read))
                tread = read

            elif CDSstop:
                center = transcript.utr3
                read = tread[max([0,center-left]):min([center+right,L])]
                if read.size!=(left+right):
                    read = np.hstack((read,-1*np.ones((left+right-read.size,),dtype=np.uint16)))
                tread = read

            elif CDScenter:
                center = (transcript.utr5+transcript.utr3)/2+np.random.randint(0,3)
                read = tread[max([0,center-left]):min([center+right,L])]
                tread = read

            reads.append(tread)

        return reads

    def get_transcript_rpkm(self, transcripts, full=True):

        rpkm = self.get_read(transcripts)
        if full:
            rpkm = [r.sum()/float(t.mask.sum())/self.total*1.e9 for r,t in zip(rpkm,transcripts)]
        else:
            rpkm = [r[t.utr5:t.utr3].sum()/float(t.utr3-t.utr5)/self.total*1.e9 for r,t in zip(rpkm,transcripts)]
        rpkm = np.array(rpkm)
        return rpkm

    def get_exon_totals(self, transcripts):

        reads = []
        for transcript in transcripts:
            if transcript.strand=='+':
                read = self.forward[transcript.chromosome][transcript.start:transcript.stop]
            else:
                read = self.reverse[transcript.chromosome][transcript.start:transcript.stop]
            counts = np.array([read[start+exten:end+exten].sum() for start,end in transcript.exons])
            #counts[0] += read[:exten].sum()
            #counts[-1] += read[-exten:].sum()
            reads.append(counts)

        return reads

    def close(self):

        self._fwd_track.close()
        self._rev_track.close()

class RnaSeq():

    def __init__(self, sample=None):

        self._track = gdb.open_track(sample)

        self.data = dict()

        for chromosome in chromosomes:
            self.data[chromosome.name] = self._track.get_array(chromosome.name)
        self.total = gdb.get_track_stat(self._track).sum

    def get_transcript_reads(self, transcripts):

        reads = []
        for transcript in transcripts:
            exons = []
            edges = list(np.where(np.logical_xor(transcript.mask[:-1],transcript.mask[1:]))[0]+1)
            edges.insert(0,0)
            edges.append(transcript.mask.size)
            exons = [(edges[i],edges[i+1]) for i in xrange(0,len(edges),2)]
            read = self.data[transcript.chromosome][transcript.start:transcript.stop]
            if transcript.strand=='-':
                read = read[::-1]
            exonread = [read[e[0]:e[1]] for e in exons]
            #read = np.hstack([np.max([1,e.sum()])*1.e6/float(e.size)/self.total*np.ones(e.shape) for e in exonread])
            read = np.hstack(exonread)
            read = np.max([1,read.sum()])*1.e6/float(read.size)/self.total #*np.ones(read.shape)

            reads.append(read)

        return reads

    def get_transcript_rpkm(self, transcripts, full=True):

        rpkm = [self.data[transcript.chromosome][transcript.start:transcript.stop][transcript.mask] \
            for transcript in transcripts]
        rpkm = [r.sum() for r in rpkm]
        #if full:
        #    rpkm = [max([1,r.sum()])/float(t.mask.sum())/self.total*1.e9 for r,t in zip(rpkm,transcripts)]
        #else:
        #    rpkm = [r[t.utr5:t.utr3].sum()/float(t.utr3-t.utr5)/self.total*1.e9 for r,t in zip(rpkm,transcripts)]
        rpkm = np.array(rpkm)
        return rpkm 

    def get_exon_totals(self, transcripts):

        reads = []
        for transcript in transcripts:
            read = self.data[transcript.chromosome][transcript.start:transcript.stop]
            counts = np.array([read[start+exten:end+exten].sum() for start,end in transcript.exons])
            #counts[0] += read[:exten].sum()
            #counts[-1] += read[-exten:].sum()
            reads.append(counts+1)

        return reads

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

        
