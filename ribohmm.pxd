import numpy as np
cimport numpy as np
from cpython cimport bool

cdef class Data:

    cdef public long L, R, M
    cdef public double scale
    cdef public np.ndarray obs, total, mappable, missingness_type
    cdef public np.ndarray log_probability, extra_log_probability
    cdef public dict codon_id

    cdef compute_log_probability(self, Emission emission)


cdef class Frame:
    
    cdef public np.ndarray posterior

    cdef update(self, Data datum, State state)


cdef class State:
    
    cdef public long M, S
    cdef public list best_start, best_stop
    cdef public np.ndarray max_posterior
    cdef public np.ndarray alpha, pos_first_moment, pos_cross_moment_start, pos_cross_moment_stop, likelihood

    cdef _forward_update(self, Data data, Transition transition)

    cdef _reverse_update(self, Data data, Transition transition)

    cdef decode(self, Data data, Transition transition, Emission emission, Frame frame)

    cdef public double joint_probability(self, Data data, Transition transition, np.ndarray[np.uint8_t, ndim=1] state, long frame)

    cdef public double compute_posterior(self, Data data, Transition transition, long start, long stop)


cdef class Transition:

    cdef public long S, C
    cdef public bool restrict
    cdef np.ndarray param
    cdef public dict seqparam

    cdef update(self, list data, list states, list frames)

cdef class Emission:

    cdef public long S, R
    cdef public np.ndarray periodicity, logperiodicity, rate_alpha, rate_beta, rescale

    cdef update_beta(self, list data, list states, list frames, double reltol)

    cdef np.ndarray _beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom)

    cdef np.ndarray _square_beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom)

cdef double normalize(np.ndarray[np.float64_t, ndim=1] x)

cdef tuple transition_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, bool restrict)

cdef tuple transition_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, bool restrict)

cdef tuple alpha_func_grad(np.ndarray[np.float64_t, ndim=2] xx, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, np.ndarray[np.float64_t, ndim=2] beta)

cdef tuple alpha_func_grad_hess(np.ndarray[np.float64_t, ndim=2] xx, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, np.ndarray[np.float64_t, ndim=2] beta)
