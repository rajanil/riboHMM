# riboHMM

**riboHMM** is an algorithm (written in Python2.x) for accurately inferring 
subsequences of an RNA molecule that are translated into protein, 
using data from ribosome profiling measurements and RNA-sequencing 
based expression measurements. riboHMM uses a mixture of hidden Markov models 
to partition the RNA sequence of a transcript into translated and untranslated regions, 
restricting to at most one translated protein sequence per transcript, using 
periodicity in the ribosome footprint counts that appear when the ribosome 
is translating an RNA molecule as it moves along the molecule.

This repo contains scripts to preprocess the raw data, quantify missing information
in the measurements, and run the algorithm. This document summarizes how to download 
and setup this software package and provides instructions on how to run the software
on a test dataset of transcript models and bam files for ribosome profiling and RNA-seq.

## Dependencies

riboHMM depends on
+ [Numpy](http://www.numpy.org/)
+ [Scipy](http://www.scipy.org/)
+ [Cvxopt](http://www.cvxopt.org/)
+ [Pysam](https://github.com/pysam-developers/pysam)
+ [Cython](http://cython.org/)

[Anaconda](https://www.continuum.io/downloads) is a great python 
distribution that already has these modules packaged in them.

## Obtaining and installing riboHMM

To obtain the source code from github, let us assume you want to clone this repo into a
directory named `proj`:

    mkdir ~/proj
    cd ~/proj
    git clone https://github.com/rajanil/riboHMM

To retrieve the latest code updates, you can do the following:

    cd ~/proj/riboHMM
    git fetch
    git merge origin/master

You will need to compile the source code as follows:

    cd ~/proj/riboHMM
    python setup.py build_ext --inplace

The setup will create some .c and .so (shared object) files, and
may give some warnings, which are OK and can be ignored.

## Required data

Minimum required data are:
(1) BAM file (sorted and indexed) containing mapped reads from a ribosome 
profiling + sequencing experiment, and 
(2) GTF file containing models for transcripts that are likely expressed in the relevant
cell type -- this would usually include the reference transcript models for the organism being studied.
(3) FASTA file (indexed) containing the reference genome sequence of the organism

Optional data are:
(4) BAM file (sorted and indexed) containing mapped reads from an RNA-sequencing 
experiment for the same cell type.

If matched RNA-sequencing data is available for the same cell type, transcripts
that are expressed in that cell type can be denovo assembled using one of several software
(e.g., [StringTie]()). The cell-specific GTF file returned by these methods can be used 
in lieu of the reference transcript models for that organism.

The following sections will describe the necessary workflow, using the example data in `test/`.

## Preprocessing the data

The bam files from ribosome profiling and RNA sequencing measurements need to be converted 
into Tabix files containing the counts of reads (ribosome footprints or RNA-seq reads) 
starting at each position in the genome. 

    # convert ribosome footprint profiling bam into tabix format
    # a separate tabix file is created for each strand and for footprints of different lengths
    # by default, footprints of length 28bp to 31bp are considered
    python bam_to_tbi.py --dtype riboseq test/hg19_test_riboseq.bam

    # convert rna-seq bam into tabix format
    python bam_to_tbi.py --dtype rnaseq test/hg19_test_rnaseq.bam

## Computing mappability for Ribo-Seq data

Since ribosome footprints are typically short (28-31 base pairs), footprints originating from
many positions in the transcriptome are likely to not be uniquely mappable. Thus, with 
standard parameters for mapping ribosome footprint sequencing data, a large fraction of the
transcriptome will have no footprints mapping to them due to mappability issues. While
riboHMM can be used without accounting for missing data due to mappability, we have observed
the results to be substantially more accurate when mappability is properly handled.

Given a GTF file that contains the transcriptome, mappability information (i.e., whether each
position in the transcriptome can produce a uniquely mappable ribosome footprint or not) can
be obtained in 3 steps:

(1) Build a FASTQ file with all footprints that could originate from the given transcriptome.
Given a GTF file, you can use `construct_synthetic_footprints.py` to generate this file.

    python construct_synthetic_footprints.py

    generate a fastq file with synthetic
    ribosome-protected fragments given a transcriptome

    positional arguments:
        gtf_file

        fasta_file

    optional arguments:
      --footprint_length FOOTPRINT_LENGTH

      --output_fastq_prefix OUTPUT_FASTQ_PREFIX

For the example GTF file in `test/`, you can run

    python construct_synthetic_reads.py --output_fastq_prefix test/hg19_synfootprints test/hg19_test.gtf /data/external_public/reference_genomes/hg19/hg19.fa

(2) Map synthetic footprints, using the same mapping strategy used for the original
ribosome footprint profiling data.
For the FASTQ file generated by Step 1, we map the footprints and retain uniquely
mapped reads, giving us the bam file `test/hg19_synfootprints_29.bam`.

(3) Build a mappability TABIX file, that marks whether a footprint originating from 
a given position uniquely mapped back to the same place.
Given a BAM file obtained from Step 2, you can use `compute_mappability.py` to generate this file.

    python compute_mappability.py

    generate a tabix file that marks whether a
    footprint originating from a given position

    positional arguments:
      bam_file

      mappability_file_prefix

For the example BAM file computed in Step 2, you can run

    python compute_mappability.py test/hg19_synfootprints_29.bam test/hg19_test_mappability_29

Note that this simple approach does not handle footprints spanning splice junctions properly;
however, junction-spanning footprints contribute very little to inference under the model
and is not likely to affect the accuracy of inference significantly.

## Learning the model

To learn the model parameters, you will need to execute `learn_model.py`.

To see command-line options that need to be passed to this script, you can do the following:

    python learn_model.py

    learns the parameters of riboHMM to infer translation from ribosome profiling 
    data and RNA sequence data; RNA-seq data can also be used if available

    positional arguments:
        fasta_file          FASTA file containing the genome
                            sequence of the organism

        gtf_file            GTF file containing the models of
                            RNA transcripts

        riboseq_file        prefix of BAM file from a ribosome profiling + 
                            sequencing assay. this prefix will be used to
                            find the relevant tabix files for ribosome
                            profiling

    optional arguments:
      --restarts RESTARTS   number of re-runs of the algorithm (default: 1)

      --mintol MINTOL       convergence criterion for change in 
                            per-nucleotide marginal likelihood (default: 1e-4)

      --scale_beta SCALE_BETA
                            scaling factor for initial precision
                            values (default: 1e4).

      --batch BATCH         number of transcripts used for learning
                            model parameters (default: 1000)

      --model_file MODEL_FILE
                            file name to store the model parameters
                            (plain text file)

      --log_file LOG_FILE   file name to store some statistics of the EM 
                            algorithm (plain text file)

      --rnaseq_file RNASEQ_FILE
                            prefix of BAM file from an RNA-seq experiment. this prefix
                            will be used to identify the relevant tabix file.

      --mappability_file MAPPABILITY_FILE
                            prefix of tabix file with mappability information

As an example, we can learn the model parameters using test transcripts by running:

    python learn_model.py --rnaseq_file test/hg19_test_rnaseq --mappability_file test/hg19_mappability --log_file test/hg19_learn_model.log --model_file test/hg19_model.txt test/hg19.fa test/hg19_test.gtf test/hg19_test_riboseq

## Inferring translated sequences

To infer translated sequences, you will need to execute `infer_CDS.py`.

To see command-line options that need to be passed to this script, you can do the following:

    python infer_CDS.py

    infers the translated sequences from ribosome profiling data and 
    RNA transcript sequence data; RNA-seq data can also be used if available

    positional arguments:
      model_file            file name to containing the model parameters

      fasta_file            FASTA file containing the genome
                            sequence of the organism

      gtf_file              GTF file containing the models of
                            RNA transcripts

      riboseq_file          BAM file from a ribosome profiling +
                            sequencing assay

    optional arguments:
      --output_file OUTPUT_FILE
                            prefix of file to write out inferred
                            translated sequences for each transcript
                            model, on each strand. the file will
                            be in BED12 format.

      --rnaseq_file RNASEQ_FILE
                            prefix of tabix file with counts of
                            RNA-seq reads

      --mappability_file MAPPABILITY_FILE
                            prefix of tabix file with mappability
                            information

Having learned the model parameters in the previous step, we can infer the translated sequence
for each RNA transcript by running:

    python infer_CDS.py --rnaseq_file test/hg19_test_rnaseq --mappability_file test/hg19_mappability test/hg19_model.txt test/hg19.fa test/hg19_test.gtf test/hg19_test_riboseq
