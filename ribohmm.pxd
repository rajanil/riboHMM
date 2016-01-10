import numpy as np
cimport numpy as np
from cpython cimport bool

cdef class Data:

    cdef public long L, R, M
    cdef public double scale
    cdef public np.ndarray obs, total, missing
    cdef public np.ndarray log_likelihood, extra_log_likelihood
    cdef public dict codon_id
    cdef public indices

    cdef compute_log_likelihood(self, Emission emission)


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

    cdef public double joint_likelihood(self, Data data, Transition transition, np.ndarray[np.uint8_t, ndim=1] state, long frame)

    cdef public double compute_posterior(self, Data data, Transition transition, long start, long stop)


cdef class Transition:

    cdef public long S, C
    cdef public str start, stop
    cdef np.ndarray param
    cdef public dict seqparam

    cdef update(self, list data, list states, list frames)

cdef class Emission:

    cdef public long S, R
    cdef public str start
    cdef public np.ndarray logperiodicity, rate_alpha, rate_beta

    cdef update_periodicity(self, list data, list states, list frames)

    cdef update_beta(self, list data, list states, list frames, double reltol)

cdef double normalize(np.ndarray[np.float64_t, ndim=1] x)

cdef extern from "gsl/gsl_sf_gamma.h":

    double  gsl_sf_lngamma(double x) nogil

cdef extern from "gsl/gsl_sf_psi.h":

    double  gsl_sf_psi(double x) nogil

    double  gsl_sf_psi_1(double x) nogil

cdef tuple transition_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, str start)

cdef tuple transition_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, str start)

cdef tuple alpha_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=2] beta, str start)

cdef tuple alpha_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=2] beta, str start)
