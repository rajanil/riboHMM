import numpy as np
cimport numpy as np
import cython
cimport cython
from cpython cimport bool
from numpy.ma.core import MaskedArray
from scipy.special import gammaln, digamma, polygamma
from math import log, exp
import scipy.optimize as opt
import cvxopt as cvx
from cvxopt import solvers
import utils
import time, pdb

solvers.options['maxiters'] = 300
solvers.options['show_progress'] = False
logistic = lambda x: 1./(1+np.exp(x))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double normalize(np.ndarray[np.float64_t, ndim=1] x):
    """Compute the log-sum-exp of a real-valued vector,
       avoiding numerical overflow issues.

    Arguments:

        x : numpy vector (float64)

    Returns:

        c : scalar (float64)

    """

    cdef long i, I
    cdef double c, xmax

    I = 9
    xmax = np.max(x)

    c = 0.
    for i from 0 <= i < I:
        c = c + exp(x[i]-xmax)
    c = log(c) + xmax
    return c

cdef np.ndarray[np.float64_t, ndim=1] outsum(np.ndarray[np.float64_t, ndim=2] arr):
    """Fast summation over the 0-th axis.
       Faster than numpy.sum()
    """

    cdef np.ndarray thesum
    thesum = sum([a for a in arr])
    return thesum

cdef class Data:

    def __cinit__(self, np.ndarray[np.uint64_t, ndim=2] obs, dict codon_id, \
                  double scale, np.ndarray[np.uint8_t, ndim=2, cast=True] mappable):
        """Instantiates a data object for a transcript and populates it 
        with observed ribosome-profiling data, expression RPKM as its
        scaling factor, and the DNA sequence of each triplet in the 
        transcript in each frame and the missingness-type for each 
        triplet in the transcript in each frame.

        """

        cdef double r,m,f
        cdef np.ndarray ma, mapp

        # length of transcript
        self.L = obs.shape[0]
        # length of HMM
        self.M = self.L/3-1
        # number of distinct footprint lengths
        self.R = obs.shape[1]
        # observed ribosome-profiling data
        self.obs = obs
        # transcript expression RPKM as scaling factor
        self.scale = scale
        # mappability of each position for each footprint length
        self.mappable = mappable
        # ensure that all unmappable positions have a zero footprint count
        self.obs[~self.mappable] = 0
        # codon type of each triplet in each frame
        self.codon_id = codon_id
        # missingness-type of each triplet in each frame for each footprint length
        self.missingness_type = np.zeros((self.R,3,self.M), dtype=np.uint8)
        # total footprint count for each triplet in each frame for each footprint length
        self.total = np.empty((3,self.M,self.R), dtype=np.uint64)
        for f from 0 <= f < 3:
            for r from 0 <= r < self.R:
                mapp = self.mappable[f:3*self.M+f,r].reshape(self.M,3)
                # missingness pattern of a triplet can belong to one of 8 types,
                # depending on which of the 3 positions are unmappable
                self.missingness_type[r,f,:] = np.array([utils.debinarize[ma.tostring()] for ma in mapp]).astype(np.uint8)
                self.total[f,:,r] = np.sum(self.obs[f:3*self.M+f,r].reshape(self.M,3),1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef compute_log_probability(self, Emission emission):
        """Computes the log probability of the data given model parameters.
           Log probability of data is the sum of log probability at positions
           under the HMM and log probability at extra positions.

           Arguments:
               emission : instance of `Emission`
                          containing estimates of 
                          emission model parameters

        """

        cdef long r, f, m, l
        cdef np.ndarray total, mask, misstypes, dat, mapA, mapB, mapAB
        cdef np.ndarray[np.float64_t, ndim=1] alpha, beta
        cdef np.ndarray[np.float64_t, ndim=2] rescale, log_probability, rate_log_probability

        self.log_probability = np.zeros((3, self.M, emission.S), dtype=np.float64)
        self.extra_log_probability = np.zeros((3,), dtype=np.float64)
        # missingness-types where 1 out of 3 positions are unmappable
        misstypes = np.array([3,5,6,7]).reshape(4,1)

        # loop over possible frames
        for f from 0 <= f < 3:

            # loop over footprint lengths
            for r from 0 <= r < self.R:

                log_probability = np.zeros((self.M,emission.S), dtype=np.float64)
                dat = self.obs[f:3*self.M+f,r].reshape(self.M,3)

                # probability under periodicity model, accounting for mappability
                # triplets with at most 1 unmappable position
                mapAB = np.any(self.missingness_type[r,f,:]==misstypes,0)
                log_probability[mapAB,:] = (gammaln(self.total[f,mapAB,r]+1) - \
                                           np.sum(gammaln(dat[mapAB,:]+1),1)).reshape(mapAB.sum(),1) + \
                                           np.dot(dat[mapAB,:], emission.logperiodicity[r].T)
                for mtype in misstypes[:3,0]:
                    mapAB = self.missingness_type[r,f,:]==mtype
                    log_probability[mapAB,:] -= np.dot(self.total[f,mapAB,r:r+1],utils.nplog(emission.rescale[r:r+1,:,mtype]))

                # probability under occupancy model, accounting for mappability
                alpha = emission.rate_alpha[r]
                beta = emission.rate_beta[r]
                rescale = emission.rescale[r,:,self.missingness_type[r,f,:]]
                total = self.total[f,:,r:r+1]
                rate_log_probability = alpha*beta*utils.nplog(beta) + \
                                       gammaln(alpha*beta + total) - \
                                       gammaln(alpha*beta) - \
                                       gammaln(total + 1) + \
                                       total*utils.nplog(self.scale*rescale) - \
                                       (alpha*beta + total) * utils.nplog(beta + self.scale*rescale)

                # ensure that triplets with all positions unmappable
                # do not contribute to the data probability
                mask = self.missingness_type[r,f,:]==0
                rate_log_probability[mask,:] = 0
                self.log_probability[f] += log_probability + rate_log_probability

                # likelihood of extra positions in transcript
                for l from 0 <= l < f:
                    if self.mappable[l,r]:
                        self.extra_log_probability[f] += alpha[0]*beta[0]*utils.nplog(beta[0]) - \
                                                         (alpha[0]*beta[0]+self.obs[l,r]) * utils.nplog(beta[0]+self.scale/3.) + \
                                                         gammaln(alpha[0]*beta[0]+self.obs[l,r]) - \
                                                         gammaln(alpha[0]*beta[0]) + \
                                                         self.obs[l,r]*utils.nplog(self.scale/3.) - \
                                                         gammaln(self.obs[l,r]+1)
                for l from 3*self.M+f <= l < self.L:
                    if self.mappable[l,r]:
                        self.extra_log_probability[f] += alpha[8]*beta[8]*utils.nplog(beta[8]) - \
                                                         (alpha[8]*beta[8]+self.obs[l,r]) * utils.nplog(beta[8]+self.scale/3.) + \
                                                         gammaln(alpha[8]*beta[8]+self.obs[l,r]) - \
                                                         gammaln(alpha[8]*beta[8]) + \
                                                         self.obs[l,r]*utils.nplog(self.scale/3.) - \
                                                         gammaln(self.obs[l,r]+1)

        # check for infs or nans in log likelihood
        if np.isnan(self.log_probability).any() \
        or np.isinf(self.log_probability).any():
            print "Warning: Inf/Nan in data log likelihood"
            pdb.set_trace()

        if np.isnan(self.extra_log_probability).any() \
        or np.isinf(self.extra_log_probability).any():
            print "Warning: Inf/Nan in extra log likelihood"
            pdb.set_trace()

cdef class Frame:
    
    def __cinit__(self):
        """Instantiates a frame object for a transcript and initializes
        a random posterior probability over all three frames.
        """

        self.posterior = np.random.rand(3)
        self.posterior = self.posterior/self.posterior.sum()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update(self, Data data, State state):
        """Update posterior probability over the three
        frames for a transcript.

        Arguments:
            data : instance of `Datum`

            state : instance of `State`

        """

        self.posterior = outsum(state.likelihood) + data.extra_log_probability
        self.posterior = self.posterior-self.posterior.max()
        self.posterior = np.exp(self.posterior)
        self.posterior = self.posterior/self.posterior.sum()

    def __reduce__(self):
        return (rebuild_Frame, (self.posterior,))

def rebuild_Frame(pos):
    f = Frame()
    f.posterior = pos
    return f

cdef class State:
    
    def __cinit__(self, long M):

        # number of triplets
        self.M = M
        # number of states for the HMM
        self.S = 9
        # stores the (start,stop) and posterior for the MAP state for each frame
        self.best_start = []
        self.best_stop = []
        self.max_posterior = np.empty((3,), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef _forward_update(self, Data data, Transition transition):

        cdef long f, i, s, m
        cdef double L, p, q
        cdef np.ndarray[np.uint8_t, ndim=1] swapidx
        cdef np.ndarray[np.float64_t, ndim=1] newalpha, logprior
        cdef np.ndarray[np.float64_t, ndim=2] P, Q

        logprior = utils.nplog([1,0,0,0,0,0,0,0,0])
        swapidx = np.array([2,3,6,7]).astype(np.uint8)
        self.alpha = np.zeros((3,self.M,self.S), dtype=np.float64)
        self.likelihood = np.zeros((self.M,3), dtype=np.float64)

        P = logistic(-1*(transition.seqparam['kozak'] * data.codon_id['kozak'] \
            + transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        for f from 0 <= f < 3:

            newalpha = logprior + data.log_probability[f,0,:]
            L = normalize(newalpha)
            for s from 0 <= s < self.S:
                self.alpha[f,0,s] = newalpha[s] - L
            self.likelihood[0,f] = L

            for m from 1 <= m < self.M:
      
                # states 2,3,6,7
                for s in swapidx:
                    newalpha[s] = self.alpha[f,m-1,s-1] + data.log_probability[f,m,s]

                # state 0,1
                try:
                    p = self.alpha[f,m-1,0] + log(1-P[m,f])
                    q = self.alpha[f,m-1,0] + log(P[m,f])
                except ValueError:
                    if P[m,f]==0.0:
                        p = self.alpha[f,m-1,0]
                        q = utils.MIN
                    else:
                        p = utils.MIN
                        q = self.alpha[f,m-1,0]
                newalpha[0] = p + data.log_probability[f,m,0]
                newalpha[1] = q + data.log_probability[f,m,1]

                # state 4
                p = self.alpha[f,m-1,3]
                try:
                    q = self.alpha[f,m-1,4] + log(1-Q[m,f])
                except ValueError:
                    q = utils.MIN
                if p>q:
                    newalpha[4] = log(1+exp(q-p)) + p + data.log_probability[f,m,4]
                else:
                    newalpha[4] = log(1+exp(p-q)) + q + data.log_probability[f,m,4]

                # state 5
                try:
                    newalpha[5] = self.alpha[f,m-1,4] + log(Q[m,f]) + data.log_probability[f,m,5]
                except ValueError:
                    newalpha[5] = utils.MIN

                # state 8
                p = self.alpha[f,m-1,7]
                q = self.alpha[f,m-1,8]
                if p>q:
                    newalpha[8] = log(1+exp(q-p)) + p + data.log_probability[f,m,8]
                else:
                    newalpha[8] = log(1+exp(p-q)) + q + data.log_probability[f,m,8]

                L = normalize(newalpha)
                for s from 0 <= s < self.S:
                    self.alpha[f,m,s] = newalpha[s] - L

                self.likelihood[m,f] = L

        if np.isnan(self.alpha).any() or np.isinf(self.alpha).any():
            print "Warning: Inf/Nan in forward update step"
            pdb.set_trace()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef _reverse_update(self, Data data, Transition transition):

        cdef long f, id, s, m
        cdef double p, q, a
        cdef np.ndarray[np.uint8_t, ndim=1] swapidx
        cdef np.ndarray[np.float64_t, ndim=1] beta, newbeta
        cdef np.ndarray[np.float64_t, ndim=2] P, Q

        swapidx = np.array([1,2,3,5,6,7]).astype(np.uint8)
        self.pos_first_moment = np.empty((3,self.M,self.S), dtype=np.float64)
        self.pos_cross_moment_start = np.empty((3,self.M,2), dtype=np.float64)

        P = logistic(-1*(transition.seqparam['kozak'] * data.codon_id['kozak'] \
            + transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        for f from 0 <= f < 3:

            self.pos_first_moment[f,self.M-1,:] = np.exp(self.alpha[f,self.M-1,:])
            newbeta = np.empty((self.S,), dtype=np.float64)
            beta = np.zeros((self.S,), dtype=np.float64)

            for m in xrange(self.M-2,-1,-1):

                for s from 0 <= s < self.S:
                    beta[s] = beta[s] + data.log_probability[f,m+1,s]

                try:
                    pp = beta[0] + log(1-P[m+1,f])
                except ValueError:
                    pp = utils.MIN
                try:
                    p = beta[1] + log(P[m+1,f])
                except ValueError:
                    p = utils.MIN
                try:
                    q = beta[5] + log(Q[m+1,f])
                except ValueError:
                    q = utils.MIN
                try:
                    qq = beta[4] + log(1-Q[m+1,f])
                except ValueError:
                    qq = utils.MIN

                # pos cross moment at start
                a = self.alpha[f,m,0] - self.likelihood[m+1,f]
                self.pos_cross_moment_start[f,m+1,0] = exp(a+p)
                self.pos_cross_moment_start[f,m+1,1] = exp(a+pp)
    
                # states 1,2,3,5,6,7
                for s in swapidx:
                    newbeta[s] = beta[s+1]
                newbeta[self.S-1] = beta[self.S-1]

                # state 0
                if p>pp:
                    newbeta[0] = log(1+np.exp(pp-p)) + p
                else:
                    newbeta[0] = log(1+np.exp(p-pp)) + pp

                # state 4
                if qq>q:
                    newbeta[4] = log(1+np.exp(q-qq)) + qq
                else:
                    newbeta[4] = log(1+np.exp(qq-q)) + q

                for s from 0 <= s < self.S:
                    beta[s] = newbeta[s] - self.likelihood[m+1,f]
                    self.pos_first_moment[f,m,s] = exp(self.alpha[f,m,s] + beta[s])

            self.pos_cross_moment_start[f,0,0] = 0
            self.pos_cross_moment_start[f,0,1] = 0

        if np.isnan(self.pos_first_moment).any() \
        or np.isinf(self.pos_first_moment).any():
            print "Warning: Inf/Nan in first moment"
            pdb.set_trace()

        if np.isnan(self.pos_cross_moment_start).any() \
        or np.isinf(self.pos_cross_moment_start).any():
            print "Warning: Inf/Nan in start cross moment"
            pdb.set_trace()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef decode(self, Data data, Transition transition, Emission emission, Frame frame):

        cdef long f, s, m
        cdef double p, q, like
        cdef np.ndarray[np.uint8_t, ndim=1] swapidx, state
        cdef np.ndarray[np.uint8_t, ndim=2] pointer
        cdef np.ndarray[np.float64_t, ndim=1] alpha, logprior
        cdef np.ndarray[np.float64_t, ndim=2] P, Q

        P = logistic(-1*(transition.seqparam['kozak'] * data.codon_id['kozak'] \
            + transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        logprior = utils.nplog([1,0,0,0,0,0,0,0,0])
        swapidx = np.array([2,3,6,7]).astype(np.uint8)
        pointer = np.zeros((self.M,self.S), dtype=np.uint8)
        pointer[0,0] = np.array([0])
        alpha = np.zeros((self.S,), dtype=np.float64)
        newalpha = np.zeros((self.S,), dtype=np.float64)
        state = np.zeros((self.M,), dtype=np.uint8)

        for f from 0 <= f < 3:

            # find the state sequence with highest posterior
            alpha = logprior + data.log_probability[f,0,:]

            for m from 1 <= m < self.M:

                # states 2,3,6,7
                for s in swapidx:
                    newalpha[s] = alpha[s-1]
                    pointer[m,s] = s-1

                # state 0,1
                try:
                    p = alpha[0] + log(1-P[m,f])
                    q = alpha[0] + log(P[m,f])
                except ValueError:
                    if P[m,f]==0.0:
                        p = alpha[0]
                        q = utils.MIN
                    else:
                        p = utils.MIN
                        q = alpha[0]
                pointer[m,0] = 0
                newalpha[0] = p
                pointer[m,1] = 0
                newalpha[1] = q

                # state 4
                p = alpha[3]
                try:
                    q = alpha[4] + log(1-Q[m,f])
                except ValueError:
                    q = utils.MIN
                if p>=q:
                    newalpha[4] = p
                    pointer[m,4] = 3
                else:
                    newalpha[4] = q
                    pointer[m,4] = 4

                # state 5
                try:
                    newalpha[5] = alpha[4] + log(Q[m,f])
                except ValueError:
                    newalpha[5] = utils.MIN
                pointer[m,5] = 4

                # state 8
                p = alpha[7]
                q = alpha[8]
                if p>=q:
                    newalpha[8] = p
                    pointer[m,8] = 7
                else:
                    newalpha[8] = q
                    pointer[m,8] = 8

                for s from 0 <= s < self.S:
                    alpha[s] = newalpha[s] + data.log_probability[f,m,s]

            # constructing the MAP state sequence
            state[self.M-1] = np.argmax(alpha)
            for m in xrange(self.M-2,0,-1):
                state[m] = pointer[m+1,state[m+1]]
            state[0] = pointer[0,0]
            self.max_posterior[f] = exp(np.max(alpha) - np.sum(self.likelihood[:,f]))

            # identifying start codon position
            try:
                self.best_start.append(np.where(state==2)[0][0]*3+f)
            except IndexError:
                self.best_start.append(None)

            # identifying stop codon position
            try:
                self.best_stop.append(np.where(state==7)[0][0]*3+f)
            except IndexError:
                self.best_stop.append(None)

        self.alpha = np.empty((1,1,1), dtype=np.float64)
        self.pos_cross_moment_start = np.empty((1,1,1), dtype=np.float64)
        self.pos_cross_moment_stop = np.empty((1,1,1), dtype=np.float64)
        self.pos_first_moment = np.empty((1,1,1), dtype=np.float64)
        self.likelihood = np.empty((1,1), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double joint_probability(self, Data data, Transition transition, 
                                  np.ndarray[np.uint8_t, ndim=1] state, long frame):

        cdef long m
        cdef double p, q, joint_probability

        joint_probability = data.log_probability[frame,0,state[0]]

        for m from 1 <= m < self.M:

            if state[m-1]==0:
    
                p = transition.seqparam['kozak'] * data.codon_id['kozak'][m,frame] \
                    + transition.seqparam['start'][data.codon_id['start'][m,frame]]
                try:
                    joint_probability = joint_probability - log(1+exp(-p))
                    if state[m]==0:
                        joint_probability = joint_probability - p
                except OverflowError:
                    if state[m]==1:
                        joint_probability = joint_probability - p

            elif state[m-1]==4:

                q = transition.seqparam['stop'][data.codon_id['stop'][m,frame]]
                try:
                    joint_probability = joint_probability - log(1+exp(-q))
                    if state[m]==4:
                        joint_probability = joint_probability - q
                except OverflowError:
                    if state[m]==5:
                        joint_probability = joint_probability - q

            joint_probability = joint_probability + data.log_probability[frame,m,state[m]]

        return joint_probability

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double compute_posterior(self, Data data, Transition transition, long start, long stop):

        cdef long frame
        cdef double joint_prob, marginal_prob, posterior
        cdef np.ndarray state

        frame = start%3
        start = (start-frame)/3
        stop = (stop-frame)/3

        # construct state sequence given a start/stop pair
        state = np.empty((self.M,), dtype=np.uint8)
        state[:start-1] = 0
        state[start-1] = 1
        state[start] = 2
        state[start+1] = 3
        state[start+2:stop-2] = 4
        state[stop-2] = 5
        state[stop-1] = 6
        state[stop] = 7
        state[stop+1:] = 8

        # compute joint probability
        joint_prob = self.joint_probability(data, transition, state, frame)
        # compute marginal probability
        marginal_prob = np.sum(self.likelihood[:,frame])
        posterior = exp(joint_prob - marginal_prob)

        return posterior

    def __reduce__(self):
        return (rebuild_State, (self.best_start, self.best_stop, self.max_posterior, self.M))

def rebuild_State(bstart, bstop, mposterior, M):
    s = State(M)
    s.best_start = bstart
    s.best_stop = bstop
    s.max_posterior = mposterior
    return s

cdef class Transition:

    def __cinit__(self):
        """Order of the states is
        '5UTS','5UTS+','TIS','TIS+','TES','TTS-','TTS','3UTS-','3UTS'
        """

        # number of states in HMM
        self.S = 9
        self.restrict = True
        self.C = len(utils.STARTCODONS)+1

        self.seqparam = dict()
        # initialize parameters for translation initiation
        self.seqparam['kozak'] = np.random.rand()
        self.seqparam['start'] = np.zeros((self.C,), dtype='float')
        self.seqparam['start'][0] = utils.MIN
        self.seqparam['start'][1] = 1+np.random.rand()
        self.seqparam['start'][2:] = utils.MIN

        # initialize parameters for translation termination
        self.seqparam['stop'] = utils.MAX*np.ones((4,), dtype=np.float64)
        self.seqparam['stop'][0] = utils.MIN

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update(self, list data, list states, list frames):

        cdef bool optimized
        cdef long id, f, V
        cdef double p, q
        cdef np.ndarray[np.float64_t, ndim=1] xo, x_final
        cdef np.ndarray[np.float64_t, ndim=2] x_init
        cdef State state
        cdef Frame frame

        # update 5'UTS -> 5'UTS+ transition parameter
        # warm start for the optimization
        optimized = False
        if self.restrict:
            xo = np.hstack((self.seqparam['kozak'],self.seqparam['start'][1:2]))
        else:
            xo = np.hstack((self.seqparam['kozak'],self.seqparam['start'][1:]))

        V = xo.size
        x_init = xo.reshape(V,1)

        try:
            x_final, optimized = optimize_transition_initiation(x_init, data, states, frames, self.restrict)
            if optimized:
                self.seqparam['kozak'] = x_final[0]
                self.seqparam['start'][1] = x_final[1]
                if not self.restrict:
                    self.seqparam['start'][2:] = x_final[2:]

        except:
            # if any error is thrown, skip updating at this iteration
            pass

    def __reduce__(self):
        return (rebuild_Transition, (self.seqparam,self.restrict))

def rebuild_Transition(seqparam, restrict):
    t = Transition()
    t.seqparam = seqparam
    t.restrict = restrict
    return t

def optimize_transition_initiation(x_init, data, states, frames, restrict):

    def func(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:
            # compute likelihood function and gradient
            results = transition_func_grad(xx, data, states, frames, restrict)
            fd = results[0]
            Df = results[1]

            # check for infs and nans, in function and gradient
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:
            # compute likelihood function, gradient and hessian
            results = transition_func_grad_hess(xx, data, states, frames, restrict)
            fd = results[0]
            Df = results[1]
            hess = results[2]

            # check for infs and nans, in function and gradient
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            # check if hessian is positive semi-definite
            eigs = np.linalg.eig(hess)
            if np.any(eigs[0]<0):
                raise ValueError
            hess = z[0] * hess
            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    # call an unconstrained nonlinear solver
    solution = solvers.cp(func)

    # check if optimal value has been reached;
    # if not, re-optimize with a cold start
    allowed = ['optimal','unknown']
    if solution['status'] in allowed:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple transition_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, bool restrict):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long j, v, V
    cdef double func, f
    cdef np.ndarray arg, df, vec, tmp, xex

    xex = np.zeros((len(utils.STARTCODONS)+1,), dtype=np.float64)
    xex[0] = utils.MIN
    xex[1] = x[1]
    if restrict:
        xex[2:] = utils.MIN
    else:
        xex[2:] = x[2:]

    V = x.size
    func = 0
    df = np.zeros((V,), dtype=float)
    for datum,state,frame in zip(data,states,frames):

        for j from 0 <= j < 3:

            arg = x[0]*datum.codon_id['kozak'][1:,j] + xex[datum.codon_id['start'][1:,j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0] * arg \
                - state.pos_cross_moment_start[j].sum(1)[1:] * utils.nplog(1+np.exp(arg)))

            # evaluate gradient
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:] * logistic(-arg)
            df[0] += frame.posterior[j] * np.sum(vec*datum.codon_id['kozak'][1:,j])
            for v from 1 <= v < V:
                tmp = datum.codon_id['start'][1:,j]==v
                df[v] += frame.posterior[j] * np.sum(vec[tmp])

    return -1.*func, -1.*df

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple transition_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, bool restrict):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long j, v, V
    cdef double func
    cdef np.ndarray xex, df, Hf, arg, vec, vec2, tmp

    xex = np.zeros((len(utils.STARTCODONS)+1,), dtype=np.float64)
    xex[0] = utils.MIN
    xex[1] = x[1]
    if restrict:
        xex[2:] = utils.MIN
    else:
        xex[2:] = x[2:]

    V = x.size
    func = 0
    df = np.zeros((V,), dtype=float)
    Hf = np.zeros((V,V), dtype=float)

    for datum,state,frame in zip(data,states,frames):

        for j from 0 <= j < 3:

            arg = x[0]*datum.codon_id['kozak'][1:,j] + xex[datum.codon_id['start'][1:,j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0] * arg \
                - state.pos_cross_moment_start[j].sum(1)[1:] * utils.nplog(1+np.exp(arg)))

            # evaluate gradient and hessian
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:] * logistic(-arg)
            vec2 = state.pos_cross_moment_start[j].sum(1)[1:] * logistic(arg) * logistic(-arg)
            df[0] += frame.posterior[j] * np.sum(vec*datum.codon_id['kozak'][1:,j])
            Hf[0,0] += frame.posterior[j] * np.sum(vec2*datum.codon_id['kozak'][1:,j]**2)
            for v from 1 <= v < V:
                tmp = datum.codon_id['start'][1:,j]==v
                df[v] += frame.posterior[j] * np.sum(vec[tmp])
                Hf[v,v] += frame.posterior[j] * np.sum(vec2[tmp])
                Hf[0,v] += frame.posterior[j] * np.sum(vec2[tmp] * datum.codon_id['kozak'][1:,j][tmp])

    Hf[:,0] = Hf[0,:]
    return -1.*func, -1.*df, Hf

cdef class Emission:

    def __cinit__(self, double scale_beta=10000.):

        cdef long r
        cdef np.ndarray[np.float64_t, ndim=1] alpha_pattern
        cdef np.ndarray[np.float64_t, ndim=2] periodicity

        self.S = 9
        self.R = len(utils.READ_LENGTHS)
        self.periodicity = np.empty((self.R,self.S,3), dtype=np.float64)
        self.logperiodicity = np.empty((self.R,self.S,3), dtype=np.float64)
        self.rescale = np.empty((self.R,self.S,8), dtype=np.float64)
        self.rate_alpha = np.empty((self.R,self.S), dtype=np.float64)
        self.rate_beta = np.empty((self.R,self.S), dtype=np.float64)
        alpha_pattern = np.array([20.,100.,1000.,100.,50.,30.,60.,10.,1.])

        for r from 0 <= r < self.R:

            periodicity = np.ones((self.S,3), dtype=np.float64)
            periodicity[1:self.S-1,:] = np.random.rand(self.S-2,1)
            self.periodicity[r] = periodicity/utils.insum(periodicity,[1])
            self.logperiodicity[r] = utils.nplog(self.periodicity[r])

            self.rate_alpha[r] = alpha_pattern*np.exp(np.random.normal(0,0.01,self.S))
            self.rate_beta[r] = scale_beta*np.random.rand(self.S)

        self.compute_rescaling()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_periodicity(self, data, states, frames):

        cdef bool optimized
        cdef long j, r, s, t, f, m, T
        cdef double ab
        cdef list constants
        cdef np.ndarray At, Bt, Ct, Et, result, argA, index
        cdef Data datum
        cdef State state
        cdef Frame frame

        T = len(data)
        Et = np.array([d.scale for d in data]).reshape(T,1)
        index = np.array([[4,5,6,7],[2,3,6,7],[1,3,5,7]]).T

        for r from 0 <= r < self.R:

            for s from 1 <= s < self.S-1:
            
                # compute constants
                At = np.zeros((3,), dtype=np.float64)
                for j in xrange(3):
                    At[j] = np.sum([frame.posterior[f] * np.sum([state.pos_first_moment[f,m,s]*datum.obs[3*m+f+j,r] 
                            for m in np.where(np.any(datum.missingness_type[r,f,:]==index[:,j:j+1],0))[0]])
                            for datum,state,frame in zip(data,states,frames) for f in xrange(3)])
  
                Bt = np.zeros((T,3), dtype=np.float64)
                Ct = np.zeros((T,3), dtype=np.float64)
                for t,(datum,state,frame) in enumerate(zip(data,states,frames)):
 
                    argA = np.array([frame.posterior[f] * state.pos_first_moment[f,:,s] *
                                     (datum.total[f,:,r]+self.rate_alpha[r,s]*self.rate_beta[r,s])
                                     for f in xrange(3)])
 
                    Bt[t,0] += np.sum(argA[datum.missingness_type[r]==3])
                    Bt[t,1] += np.sum(argA[datum.missingness_type[r]==5])
                    Bt[t,2] += np.sum(argA[datum.missingness_type[r]==6])
                    Ct[t,0] += np.sum(argA[datum.missingness_type[r]==4])
                    Ct[t,1] += np.sum(argA[datum.missingness_type[r]==2])
                    Ct[t,2] += np.sum(argA[datum.missingness_type[r]==1])

                constants = [At, Bt, Ct, Et, self.rate_beta[r,s]]

                # run optimizer
                try:
                    result, optimized = optimize_periodicity(self.periodicity[r,s,:], constants)
                    if optimized:
                        result[result<=0] = utils.EPS
                        result = result/result.sum()
                        self.periodicity[r,s,:] = result
                except:
                    # if any error is thrown, skip updating at this iteration
                    pass

        self.logperiodicity = utils.nplog(self.periodicity)

        if np.isinf(self.logperiodicity).any() or np.isnan(self.logperiodicity).any():
            print "Warning: Inf/Nan in periodicity parameter"

        self.compute_rescaling()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def compute_rescaling(self):

        cdef long r, s, j
        cdef np.ndarray mask

        for r in xrange(self.R):

            for s in xrange(self.S):

                for j,mask in utils.binarize.iteritems():
                
                    self.rescale[r,s,j] = np.sum(self.periodicity[r,s,mask])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update_beta(self, list data, list states, list frames, double reltol):

        cdef Data datum
        cdef State state
        cdef Frame frame
        cdef long r, s, f
        cdef double diff, reldiff
        cdef np.ndarray beta, newbeta, denom, new_scale, mask

        reldiff = 1
        newbeta = self.rate_beta.copy()

        denom = np.zeros((self.R, self.S), dtype=np.float64)
        for datum,state,frame in zip(data,states,frames):

            for r in xrange(self.R):

                for s in xrange(self.S):

                    new_scale = self.rescale[r,s,datum.missingness_type[r]]
                    mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                    if np.all(mask):
                        continue
                    denom[r,s] = denom[r,s] + np.sum(frame.posterior * \
                                 np.sum(MaskedArray(state.pos_first_moment[:,:,s], mask=mask),1))

                denom[r,0] = denom[r,0] + np.sum(frame.posterior * \
                             np.array([datum.mappable[:f,r].sum() for f in xrange(3)]))
                denom[r,8] = denom[r,8] + np.sum(frame.posterior * \
                             np.array([datum.mappable[3*datum.M+f:,r].sum() for f in xrange(3)]))

        while reldiff>reltol:

            beta = newbeta.copy()
            newbeta = self._square_beta_map(beta, data, states, frames, denom)
            diff = np.abs(newbeta-beta).sum()
            reldiff = np.mean(np.abs(newbeta-beta)/beta)

        self.rate_beta = newbeta.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray _square_beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom):

        cdef int step
        cdef bool a_ok
        cdef np.ndarray R, V, a
        cdef list vars

        vars = [beta]
        for step in [0,1]:
            beta = self._beta_map(beta, data, states, frames, denom)
            vars.append(beta)

        R = vars[1] - vars[0]
        V = vars[2] - vars[1] - R
        a = -1.*np.sqrt(R**2/V**2)
        a[a>-1] = -1.
        a[np.logical_or(np.abs(R)<1e-4,np.abs(V)<1e-4)] = -1.

        # given two update steps, compute an optimal step that achieves
        # a better likelihood than the two steps.
        a_ok = False
        while not a_ok:

            beta = (1+a)**2*vars[0] - 2*a*(1+a)*vars[1] + a**2*vars[2]

            mask = beta<=0
            if np.any(mask):
                a[mask] = (a[mask]-1)/2.
                a[np.abs(a+1)<1e-4] = -1.
            else:
                a_ok = True
  
        beta = self._beta_map(beta, data, states, frames, denom)

        return beta
 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray _beta_map(self, np.ndarray beta, list data, list states, list frames, np.ndarray denom):

        cdef long f, r, s, l
        cdef np.ndarray newbeta, argA, argB, new_scale, mask, pos
        cdef Data datum
        cdef State state
        cdef Frame frame

        newbeta = np.zeros((self.R,self.S), dtype=np.float64)
        for datum,state,frame in zip(data,states,frames):

            for r in xrange(self.R):

                for s in xrange(self.S):

                    new_scale = self.rescale[r,s,datum.missingness_type[r]]
                    mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                    if np.all(mask):
                        continue

                    argA = datum.scale*MaskedArray(new_scale, mask=mask) + beta[r,s]
                    argB = MaskedArray(datum.total[:,:,r], mask=mask) + self.rate_alpha[r,s]*beta[r,s]
                    pos = MaskedArray(state.pos_first_moment[:,:,s], mask=mask)
                    newbeta[r,s] = newbeta[r,s] + np.sum(frame.posterior * np.sum(pos *
                                   (utils.nplog(argA) - digamma(argB) + argB/argA/self.rate_alpha[r,s]),1))

                for f in xrange(3):
                    # add extra terms for first state
                    for l from 0 <= l < f:
                        if datum.mappable[l,r]:
                            newbeta[r,0] = newbeta[r,0] + frame.posterior[f] * \
                                           ((utils.nplog(datum.scale/3.+beta[r,0]) - \
                                            digamma(datum.obs[l,r]+self.rate_alpha[r,0]*beta[r,0])) + \
                                            (datum.obs[l,r]+self.rate_alpha[r,0]*beta[r,0]) / \
                                            self.rate_alpha[r,0]/(datum.scale/3.+beta[r,0]))

                    # add extra terms for last state
                    for l from 3*datum.M+f <= l < datum.L:
                        if datum.mappable[l,r]:
                            newbeta[r,8] = newbeta[r,8] + frame.posterior[f] * \
                                           ((utils.nplog(datum.scale/3.+beta[r,8]) - \
                                            digamma(datum.obs[l,r]+self.rate_alpha[r,8]*beta[r,8])) + \
                                            (datum.obs[l,r]+self.rate_alpha[r,8]*beta[r,8]) / \
                                            self.rate_alpha[r,8]/(datum.scale/3.+beta[r,8]))

        newbeta = np.exp(newbeta / denom + digamma(self.rate_alpha*beta) - 1)
        return newbeta

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_alpha(self, data, states, frames):

        cdef bool optimized
        cdef long s
        cdef list constants
        cdef np.ndarray x_init, x_final

        # warm start for the optimization
        optimized = False
        x_init = self.rate_alpha

        try:
            x_final, optimized = optimize_alpha(x_init, data, states, frames, self.rescale, self.rate_beta)
            if optimized:
                self.rate_alpha = x_final

        except ValueError:
            # if any error is thrown, skip updating at this iteration
            pass

    def __reduce__(self):
        return (rebuild_Emission, (self.periodicity, self.rate_alpha, self.rate_beta))

def rebuild_Emission(periodicity, alpha, beta):
    e = Emission()
    e.periodicity = periodicity
    e.logperiodicity = utils.nplog(periodicity)
    e.rate_alpha = alpha
    e.rate_beta = beta
    e.compute_rescaling()
    return e

def optimize_periodicity(x_init, constants):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).reshape(1,3)
        At = constants[0]
        Bt = constants[1]
        Ct = constants[2]
        Et = constants[3]
        ab = constants[4]

        # compute function
        func = np.sum(At * utils.nplog(xx)) - \
               np.sum(Bt * utils.nplog((1-xx)*Et+ab)) - \
               np.sum(Ct * utils.nplog(xx*Et+ab))
        if np.isnan(func) or np.isinf(func):
            f = np.array([np.finfo(np.float32).max]).astype(np.float64)
        else:
            f = np.array([-1*func]).astype(np.float64)

        # compute gradient
        Df = At/xx[0] + np.sum(Bt * Et/((1-xx)*Et+ab),0) - \
             np.sum(Ct * Et/(xx*Et+ab),0)
        if np.isnan(Df).any() or np.isinf(Df).any():
            Df = -1 * np.finfo(np.float32).max * \
                 np.ones((1,xx.size), dtype=np.float64)
        else:
            Df = -1*Df.reshape(1,xx.size)

        if z is None:
            return cvx.matrix(f), cvx.matrix(Df)

        # compute hessian
        hess = 1.*At/xx[0]**2 - np.sum(Bt * Et**2/((1-xx)*Et+ab)**2,0) - \
               np.sum(Ct * Et**2/(xx*Et+ab)**2,0)

        # check if hessian is positive semi-definite
        if np.any(hess<0) or np.any(np.isnan(hess)) or np.any(np.isinf(hess)):
            raise ValueError
        else:
            hess = z[0] * np.diag(hess)

        return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    V = x_init.size
    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((V,), dtype=np.float64)))
    h = cvx.matrix(np.zeros((V,1), dtype=np.float64))
    A = cvx.matrix(np.ones((1,V), dtype=np.float64))
    b = cvx.matrix(np.ones((1,1), dtype=np.float64))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h, A=A, b=b)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized

def optimize_alpha(x_init, data, states, frames, rescale, beta):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init.reshape(V,1))

        xx = np.array(x).reshape(beta.shape)

        if z is None:
            # compute likelihood function and gradient
            results = alpha_func_grad(xx, data, states, frames, rescale, beta)

            # check for infs or nans
            fd = results[0]
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            Df = results[1]
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            return cvx.matrix(f), cvx.matrix(Df)

        else:
            # compute function, gradient, and hessian
            results = alpha_func_grad_hess(xx, data, states, frames, rescale, beta)

            # check for infs or nans
            fd = results[0]
            if np.isnan(fd) or np.isinf(fd):
                f = np.array([np.finfo(np.float32).max]).astype(np.float64)
            else:
                f = np.array([fd]).astype(np.float64)

            Df = results[1]
            if np.isnan(Df).any() or np.isinf(Df).any():
                Df = -1 * np.finfo(np.float32).max * np.ones((1,xx.size), dtype=np.float64)
            else:
                Df = Df.reshape(1,xx.size)

            # check if hessian is positive semi-definite
            hess = results[2]
            if np.any(hess)<0:
                raise ValueError
            hess = np.diag(z[0] * hess.ravel())

            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    V = x_init.size
    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((V,), dtype=np.float64)))
    h = cvx.matrix(np.zeros((V,1), dtype=np.float64))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).reshape(x_init.shape)
    x_final[x_final<=0] = x_init[x_final<=0]

    return x_final, optimized

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple alpha_func_grad(np.ndarray[np.float64_t, ndim=2] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, np.ndarray[np.float64_t, ndim=2] beta):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long s, f, l, r, R, S
    cdef double func
    cdef np.ndarray gradient, mask, new_scale, argA, argB, argC, pos

    R = beta.shape[0]
    S = beta.shape[1]

    func = 0
    gradient = np.zeros((R,S), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):
                    
        for r in xrange(R):

            for s in xrange(S):

                new_scale = rescale[r,s,datum.missingness_type[r]]
                mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                if np.all(mask):
                    continue

                argA = MaskedArray(datum.total[:,:,r], mask=mask, fill_value=1) + x[r,s]*beta[r,s]
                argB = datum.scale*MaskedArray(new_scale, mask=mask, fill_value=1) + beta[r,s]
                pos = MaskedArray(state.pos_first_moment[:,:,s], mask=mask, fill_value=1)
                argC = np.sum(pos, 1)

                func = func + np.sum(frame.posterior * np.sum(pos * \
                       (gammaln(argA) - argA*utils.nplog(argB)),1)) + np.sum(frame.posterior * \
                       argC) * (x[r,s]*beta[r,s]*utils.nplog(beta[r,s])-gammaln(x[r,s]*beta[r,s]))

                gradient[r,s] = gradient[r,s] + beta[r,s] * np.sum(frame.posterior * \
                                np.sum(pos * (digamma(argA) - utils.nplog(argB)), 1)) + \
                                np.sum(frame.posterior * argC) * beta[r,s] * (utils.nplog(beta[r,s]) - \
                                digamma(x[r,s]*beta[r,s]))

            for f in xrange(3):
                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,0]*beta[r,0]*utils.nplog(beta[r,0]) + 
                               gammaln(datum.obs[l,r]+x[r,0]*beta[r,0]) - gammaln(x[r,0]*beta[r,0]) - 
                               (datum.obs[l,r]+x[r,0]*beta[r,0]) * utils.nplog(datum.scale/3.+beta[r,0]))
                        gradient[r,0] = gradient[r,0] + frame.posterior[f] * beta[r,0] * (utils.nplog(beta[r,0]) + 
                                   digamma(datum.obs[l,r]+x[r,0]*beta[r,0]) - digamma(x[r,0]*beta[r,0]) - 
                                   utils.nplog(datum.scale/3.+beta[r,0]))

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,8]*beta[r,8]*utils.nplog(beta[r,8]) + 
                               gammaln(datum.obs[l,r]+x[r,8]*beta[r,8]) - gammaln(x[r,8]*beta[r,8]) - 
                               (datum.obs[l,r]+x[r,8]*beta[r,8]) * utils.nplog(datum.scale/3.+beta[r,8]))
                        gradient[r,8] = gradient[r,8] + frame.posterior[f] * beta[r,8] * (utils.nplog(beta[r,8]) + 
                                   digamma(datum.obs[l,r]+x[r,8]*beta[r,8]) - digamma(x[r,8]*beta[r,8]) - 
                                   utils.nplog(datum.scale/3.+beta[r,8]))

    func = -1.*func
    gradient = -1.*gradient

    return func, gradient

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple alpha_func_grad_hess(np.ndarray[np.float64_t, ndim=2] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, np.ndarray[np.float64_t, ndim=2] beta):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, s, f, R, S, l
    cdef double func
    cdef np.ndarray gradient, mask, new_scale, argA, argB, argC, pos, hessian

    R = beta.shape[0]
    S = beta.shape[1]

    func = 0
    gradient = np.zeros((R,S), dtype=np.float64)
    hessian = np.zeros((R,S), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):

        for r in xrange(R):

            for s in xrange(S):

                new_scale = rescale[r,s,datum.missingness_type[r]]
                mask = np.logical_not(np.logical_and(new_scale>0,state.pos_first_moment[:,:,s]>0))
                if np.all(mask):
                    continue

                argA = MaskedArray(datum.total[:,:,r], mask=mask, fill_value=1) + x[r,s]*beta[r,s]
                argB = datum.scale*MaskedArray(new_scale, mask=mask, fill_value=1) + beta[r,s]
                pos = MaskedArray(state.pos_first_moment[:,:,s], mask=mask, fill_value=1)
                argC = np.sum(pos, 1)

                func = func + np.sum(frame.posterior * np.sum(pos * \
                       (gammaln(argA) - argA*utils.nplog(argB)),1)) + np.sum(frame.posterior * \
                       argC) * (x[r,s]*beta[r,s]*utils.nplog(beta[r,s])-gammaln(x[r,s]*beta[r,s]))

                gradient[r,s] = gradient[r,s] + beta[r,s] * np.sum(frame.posterior * \
                                np.sum(pos * (digamma(argA) - utils.nplog(argB)), 1)) + \
                                np.sum(frame.posterior * argC) * (utils.nplog(beta[r,s]) - \
                                digamma(x[r,s]*beta[r,s])) * beta[r,s]
                        
                hessian[r,s] = hessian[r,s] + beta[r,s]**2 * np.sum(frame.posterior * \
                               np.sum(pos * polygamma(1,argA),1)) - beta[r,s]**2 * \
                               np.sum(frame.posterior * argC) * polygamma(1,x[r,s]*beta[r,s])

            for f in xrange(3):
                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,0]*beta[r,0]*utils.nplog(beta[r,0]) + \
                               gammaln(datum.obs[l,r]+x[r,0]*beta[r,0]) - gammaln(x[r,0]*beta[r,0]) - \
                               (datum.obs[l,r]+x[r,0]*beta[r,0]) * utils.nplog(datum.scale/3.+beta[r,0]))
                        gradient[r,0] = gradient[r,0] + frame.posterior[f] * beta[r,0] * (utils.nplog(beta[r,0]) + \
                                        digamma(datum.obs[l,r]+x[r,0]*beta[r,0]) - digamma(x[r,0]*beta[r,0]) - \
                                        utils.nplog(datum.scale/3.+beta[r,0]))
                        hessian[r,0] = hessian[r,0] + frame.posterior[f] * beta[r,0]**2 * \
                                       (polygamma(1,datum.obs[l,r]+x[r,0]*beta[r,0]) - \
                                       polygamma(1,x[r,0]*beta[r,0]))

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x[r,8]*beta[r,8]*utils.nplog(beta[r,8]) + \
                               gammaln(datum.obs[l,r]+x[r,8]*beta[r,8]) - gammaln(x[r,8]*beta[r,8]) - \
                               (datum.obs[l,r]+x[r,8]*beta[r,8]) * utils.nplog(datum.scale/3.+beta[r,8]))
                        gradient[r,8] = gradient[r,8] + frame.posterior[f] * beta[r,8] * (utils.nplog(beta[r,8]) + \
                                        digamma(datum.obs[l,r]+x[r,8]*beta[r,8]) - digamma(x[r,8]*beta[r,8]) - \
                                        utils.nplog(datum.scale/3.+beta[r,8]))
                        hessian[r,8] = hessian[r,8] + frame.posterior[f] * beta[r,8]**2 * \
                                       (polygamma(1,datum.obs[l,r]+x[r,8]*beta[r,8]) - \
                                       polygamma(1,x[r,8]*beta[r,8]))

    func = -1.*func
    gradient = -1.*gradient
    hessian = -1.*hessian

    return func, gradient, hessian

def learn_parameters(observations, codon_id, scales, mappability, scale_beta, restarts, mintol):

    cdef long restart, i, D
    cdef double scale, Lmax, L, dL, newL, reltol, starttime, totaltime
    cdef str start
    cdef list data, Ls, states, frames, ig
    cdef dict id
    cdef np.ndarray observation, mappable
    cdef Data datum
    cdef Emission emission, best_emission
    cdef Transition transition, best_transition
    cdef State state
    cdef Frame frame

    data = [Data(observation, id, scale, mappable)
               for observation, id, scale, mappable in
               zip(observations, codon_id, scales, mappability)]

    # Initialize latent variables
    states = [State(datum.M) for datum in data]
    frames = [Frame() for datum in data]

    print "Stage 1: allow only AUG start codons; only update periodicity parameters ..."
    transition = Transition()
    emission = Emission(scale_beta)

    # compute initial log likelihood
    for datum,state,frame in zip(data,states,frames):
        datum.compute_log_probability(emission)
        state._forward_update(datum, transition)
        frame.update(datum, state)
    L = np.sum([np.sum(frame.posterior*state.likelihood) \
            + np.sum(frame.posterior*datum.extra_log_probability) \
            for datum,state,frame in zip(data,states,frames)]) / \
            np.sum([datum.L for datum in data])
    dL = np.inf

    # iterate till convergence
    reltol = dL/np.abs(L)
    while np.abs(reltol)>mintol:

        starttime = time.time()

        for state,datum in zip(states, data):
            state._reverse_update(datum, transition)

        # update periodicity parameters
        emission.update_periodicity(data, states, frames)

        # update transition parameters
        transition.update(data, states, frames)

        # compute log likelihood
        for datum,state,frame in zip(data,states,frames):
            datum.compute_log_probability(emission)
            state._forward_update(datum, transition)
            frame.update(datum, state)
        newL = np.sum([np.sum(frame.posterior*state.likelihood) \
            + np.sum(frame.posterior*datum.extra_log_probability) \
            for datum,state,frame in zip(data,states,frames)]) / \
            np.sum([datum.L for datum in data])

        dL = newL-L
        L = newL
        reltol = dL/np.abs(L)
        print L, reltol, time.time()-starttime

    print "Stage 2: allow only AUG start codons; update all parameters ..."
    dL = np.inf
    reltol = dL/np.abs(L)
    while np.abs(reltol)>mintol:

        starttime = time.time()

        # update latent states
        for state,datum in zip(states, data):
            state._reverse_update(datum, transition)

        # update periodicity parameters
        emission.update_periodicity(data, states, frames)

        # update occupancy parameters
        emission.update_alpha(data, states, frames)
        emission.update_beta(data, states, frames, 1e-3)

        # update transition parameters
        transition.update(data, states, frames)

        # compute log likelihood
        for datum,state,frame in zip(data,states,frames):
            datum.compute_log_probability(emission)
            state._forward_update(datum, transition)
            frame.update(datum, state)
        newL = np.sum([np.sum(frame.posterior*state.likelihood) \
            + np.sum(frame.posterior*datum.extra_log_probability) \
            for datum,state,frame in zip(data,states,frames)]) / \
            np.sum([datum.L for datum in data])

        dL = newL-L
        L = newL
        reltol = dL/np.abs(L)
        print L, reltol, time.time()-starttime

    print "Stage 3: allow noncanonical start codons ..."
    transition.restrict = False
    transition.seqparam['start'][2:] = -3+np.random.rand(transition.seqparam['start'][2:].size)
    for datum,state,frame in zip(data,states,frames):
        state._forward_update(datum, transition)
        frame.update(datum, state)

    dL = np.inf
    reltol = dL/np.abs(L)
    while np.abs(reltol)>mintol:

        totaltime = time.time()

        # update latent variables
        starttime = time.time()
        for state,datum in zip(states, data):
            state._reverse_update(datum, transition)

        # update transition parameters for noncanonical codons
        transition.update(data, states, frames)

        # compute log likelihood
        for datum,state,frame in zip(data, states, frames):
            state._forward_update(datum, transition)
            frame.update(datum, state)
        newL = np.sum([np.sum(frame.posterior*state.likelihood) \
            + np.sum(frame.posterior*datum.extra_log_probability) \
            for datum,state,frame in zip(data,states,frames)]) / \
            np.sum([datum.L for datum in data])

        dL = newL-L
        L = newL
        reltol = dL/np.abs(L)
        print L, reltol, time.time()-starttime

    return transition, emission, L

def infer_coding_sequence(observations, codon_id, scales, mappability, transition, emission):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef dict id
    cdef double scale
    cdef list data, states, frames
    cdef np.ndarray observation, mappable 

    data = [Data(observation, id, scale, mappable) for observation,id,scale,mappable in zip(observations,codon_id,scales,mappability)]
    states = [State(datum.M) for datum in data]
    frames = [Frame() for datum in data]

    for state,frame,datum in zip(states, frames, data):
        datum.compute_log_probability(emission)
        state._forward_update(datum, transition)
        frame.update(datum, state)
        state.decode(datum, transition, emission, frame)

    return states, frames

