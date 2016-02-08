import numpy as np
cimport numpy as np

cdef class RnaSequence:

    cdef public long S
    cdef public str sequence
    
    cdef np.ndarray _mark_start_codons(self)

    cdef np.ndarray _mark_stop_codons(self)

    cdef np.ndarray _compute_kozak_scores(self)
