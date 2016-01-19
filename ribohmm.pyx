import numpy as np
cimport numpy as np
import cython
cimport cython
from cpython cimport bool
from scipy.special import gammaln, digamma, polygamma
from math import log, exp
import cvxopt as cvx
from cvxopt import solvers
import utils
import time, pdb

solvers.options['maxiters'] = 15
solvers.options['show_progress'] = False
logistic = lambda x: 1./(1+np.exp(x))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double normalize(np.ndarray[np.float64_t, ndim=1] x):

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

    """
    Fast summation over the 0-th axis.
    """

    cdef np.ndarray thesum
    thesum = sum([a for a in arr])
    return thesum

cdef class Data:

    def __cinit__(self, np.ndarray[np.uint64_t, ndim=2] obs, dict codon_id, double scale, np.ndarray[np.uint8_t, ndim=2, cast=True] mappable):

        cdef double r,m,f

        self.L = obs.shape[0]
        self.M = self.L/3-1
        self.R = obs.shape[1]
        self.obs = obs
        self.scale = scale
        self.mappable = mappable
        self.codon_id = codon_id
        self.rescale_indices = np.zeros((self.R,3,self.M), dtype=np.uint8)
        self.total = np.empty((3,self.M,self.R), dtype=np.uint64)
        for f from 0 <= f < 3:
            for r from 0 <= r < self.R:
                self.rescale_indices[r,f] = np.array([utils.debinarize(self.mappable[3*m+f:3*m+3+f,r]) 
                                            for m in xrange(self.M)]).astype(np.uint8)
                self.total[f,:,r] = np.array([self.obs[3*m+f:3*m+3+f,r][self.mappable[3*m+f:3*m+3+f,r]].sum()
                                    for m in xrange(self.M)])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef compute_log_likelihood(self, Emission emission):

        cdef long r, f, m, l
        cdef np.ndarray rescale, mask
        cdef np.ndarray[np.float64_t, ndim=2] log_likelihood, rate_log_likelihood

        self.log_likelihood = np.zeros((3, self.M, emission.S), dtype=np.float64)
        self.extra_log_likelihood = np.zeros((3,), dtype=np.float64)

        for f from 0 <= f < 3:

            for r from 0 <= r < self.R:

                log_likelihood = np.zeros((self.M,emission.S), dtype=np.float64)

                # periodicity likelihood, accounting for mappability
                log_likelihood = np.array([gammaln(self.total[f,m,r]+1) 
                    - np.sum(gammaln(self.obs[3*m+f:3*m+3+f,r]+1)) 
                    if np.all(self.mappable[3*m+f:3*m+3+f,r]) else
                    gammaln(self.total[f,m,r]+1)
                    - np.sum(gammaln(self.obs[3*m+f:3*m+3+f,r][self.mappable[3*m+f:3*m+3+f,r]]+1))
                    if np.count_nonzero(self.mappable[3*m+f:3*m+3+f,r])==2 else
                    0 for m in xrange(self.M)])
                log_likelihood += np.array([np.sum(self.obs[3*m+f:3*m+3+f,r] 
                    * emission.logperiodicity[r],1)
                    if np.all(self.mappable[3*m+f:3*m+3+f,r]) else
                    self.obs[3*m+f:3*m+3+f,r][self.mappable[3*m+f:3*m+3+f,r]]
                    * (emission.logperiodicity[r,self.mappable[3*m+f:3*m+3+f,r],:]
                    - np.log(utils.insum(emission.periodicity[r,self.mappable[3*m+f:3*m+3+f,r],:],[1])))
                    if np.count_nonzero(self.mappable[3*m+f:3*m+3+f,r])==2 else
                    0 for m in xrange(self.M)])

                if not emission.restrict:

                    # occupancy likelihood, accounting for mappability
                    mask = datum.rescale_indices[r,f,:]==0
                    rescale = emission.rescale[r,s,datum.rescale_indices[r,f,:]]
                    rate_log_likelihood = emission.rate_alpha[r] * emission.rate_beta[r] \
                        * utils.nplog(emission.rate_beta[r]) + gammaln(emission.rate_alpha[r] \
                        * emission.rate_beta[r] + self.total[f,:,r:r+1]) \
                        - gammaln(emission.rate_alpha[r] * emission.rate_beta[r]) \
                        - gammaln(self.total[f,:,r:r+1]+1)
                    for s in xrange(emission.S):
                        rescale = emission.rescale[r,s,datum.rescale_indices[r,f,:]]
                        rate_log_likelihood[:,s] = rate_log_likelihood[:,s] - (emission.rate_alpha[r,s] * 
                                                   emission.rate_beta[r,s] + self.total[f,:,r:r+1]) * 
                                                   utils.nplog(emission.rate_beta[r,s] + self.scale*rescale)
                        rate_log_likelihood[mask,s] = rate_log_likelihood[mask,s] + self.total[f,mask,r:r+1] * 
                                                      utils.nplog(rescale[mask]*self.scale)
                    rate_log_likelihood[mask] = 0
                    log_likelihood += rate_log_likelihood

                    # likelihood of extra positions
                    for l from 0 <= l < f:
                        if self.mappable[l,r]:
                            self.extra_log_likelihood[f] += emission.rate_alpha[r,0] * emission.rate_beta[r,0] \
                                * utils.nplog(emission.rate_beta[r,0]) - (emission.rate_alpha[r,0] \
                                * emission.rate_beta[r,0] + self.obs[l,r]) \
                                * utils.nplog(emission.rate_beta[r,0] + self.scale/3.) \
                                + gammaln(emission.rate_alpha[r,0] * emission.rate_beta[r,0] + self.obs[l,r]) \
                                - gammaln(emission.rate_alpha[r,0] * emission.rate_beta[r,0]) \
                                + self.obs[l,r] * utils.nplog(self.scale/3.) - gammaln(self.obs[l,r]+1)
                    for l from 3*self.M+f <= l < self.L:
                        if self.mappable[l,r]:
                            self.extra_log_likelihood[f] += emission.rate_alpha[r,emission.S-1] \
                                * emission.rate_beta[r,emission.S-1] * utils.nplog(emission.rate_beta[r,emission.S-1]) \
                                - (emission.rate_alpha[r,emission.S-1] * emission.rate_beta[r,emission.S-1] \
                                + self.obs[l,r]) * utils.nplog(emission.rate_beta[r,emission.S-1] \
                                + self.scale/3.) + gammaln(emission.rate_alpha[r,emission.S-1] \
                                * emission.rate_beta[r,emission.S-1] + self.obs[l,r]) \
                                - gammaln(emission.rate_alpha[r,emission.S-1] * emission.rate_beta[r,emission.S-1]) \
                                + self.obs[l,r] * utils.nplog(self.scale/3.) - gammaln(self.obs[l,r]+1)

                self.log_likelihood[f] += log_likelihood
   
        # check for infs or nans in log likelihood
        if np.isnan(self.log_likelihood).any() \
        or np.isinf(self.log_likelihood).any():
            print "Warning: Inf/Nan in data log likelihood"
            pdb.set_trace()

        if np.isnan(self.extra_log_likelihood).any() \
        or np.isinf(self.extra_log_likelihood).any():
            print "Warning: Inf/Nan in extra log likelihood"
            pdb.set_trace()

cdef class Frame:
    
    def __cinit__(self):

        self.posterior = np.random.rand(3)
        self.posterior = self.posterior/self.posterior.sum()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update(self, Data data, State state):

        self.posterior = outsum(state.likelihood) + data.extra_log_likelihood
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
        self.alpha = np.empty((3,self.M,self.S), dtype=np.float64)
        self.likelihood = np.empty((self.M,3), dtype=np.float64)
        newalpha = np.empty((self.S,), dtype=np.float64)

        if transition.restrict:
            P = logistic(-1*transition.seqparam['start'][data.codon_id['start']])
        else:
            P = logistic(-1*(transition.seqparam['kozak'] * data.codon_id['kozak'] \
                + transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        for f from 0 <= f < 3:

            newalpha = logprior + data.log_likelihood[f,0,:]
            L = normalize(newalpha)
            for s from 0 <= s < self.S:
                self.alpha[f,0,s] = newalpha[s] - L
            self.likelihood[0,f] = L

            for m from 1 <= m < self.M:
      
                # states 2,3,6,7
                for s in swapidx:
                    newalpha[s] = self.alpha[f,m-1,s-1] + data.log_likelihood[f,m,s]

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
                newalpha[0] = p + data.log_likelihood[f,m,0]
                newalpha[1] = q + data.log_likelihood[f,m,1]

                # state 4
                p = self.alpha[f,m-1,3]
                try:
                    q = self.alpha[f,m-1,4] + log(1-Q[m,f])
                except ValueError:
                    q = utils.MIN
                if p>q:
                    newalpha[4] = log(1+exp(q-p)) + p + data.log_likelihood[f,m,4]
                else:
                    newalpha[4] = log(1+exp(p-q)) + q + data.log_likelihood[f,m,4]

                # state 5
                try:
                    newalpha[5] = self.alpha[f,m-1,4] + log(Q[m,f]) + data.log_likelihood[f,m,5]
                except ValueError:
                    newalpha[5] = utils.MIN

                # state 8
                p = self.alpha[f,m-1,7]
                q = self.alpha[f,m-1,8]
                if p>q:
                    newalpha[8] = log(1+exp(q-p)) + p + data.log_likelihood[f,m,8]
                else:
                    newalpha[8] = log(1+exp(p-q)) + q + data.log_likelihood[f,m,8]

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

        if transition.restrict:
            P = logistic(-1*transition.seqparam['start'][data.codon_id['start']])
        else:
            P = logistic(-1*(transition.seqparam['kozak'] * data.codon_id['kozak'] \
                + transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        for f from 0 <= f < 3:

            self.pos_first_moment[f,self.M-1,:] = np.exp(self.alpha[f,self.M-1,:])
            newbeta = np.empty((self.S,), dtype=np.float64)
            beta = np.zeros((self.S,), dtype=np.float64)

            for m in xrange(self.M-2,-1,-1):

                for s from 0 <= s < self.S:
                    beta[s] = beta[s] + data.log_likelihood[f,m+1,s]

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
        pointer = np.empty((self.M,self.S), dtype=np.uint8)
        pointer[0,0] = np.array([0])
        alpha = np.empty((self.S,), dtype=np.float64)
        newalpha = np.empty((self.S,), dtype=np.float64)
        state = np.zeros((self.M,), dtype=np.uint8)

        for f from 0 <= f < 3:

            # find the state sequence with highest posterior
            alpha = logprior + data.log_likelihood[f,0,:]

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
                    alpha[s] = newalpha[s] + data.log_likelihood[f,m,s]

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

        joint_probability = data.log_likelihood[frame,0,state[0]]

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

            joint_probability = joint_probability + data.log_likelihood[frame,m,state[m]]

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
        self.C = len(set(utils.STARTS.values()))+1

        self.seqparam = dict()
        # initialize parameters for translation initiation
        self.seqparam['kozak'] = 0
        self.seqparam['start'] = -1*np.random.rand(self.C)
        self.seqparam['start'][0] = utils.MIN
        self.seqparam['start'][1] = 1+np.random.rand()

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
            xo = self.seqparam['start'][1:2]
        else:
            xo = np.hstack((self.seqparam['kozak'],self.seqparam['start'][1:]))

        V = xo.size
        x_init = xo.reshape(V,1)

        while not optimized:
            try:
                x_final, optimized = optimize_transition_initiation(x_init, data, states, frames, self.restrict)
                if not optimized:
                    # retry optimization with near cold start
                    x_init = x_init*np.exp(np.random.normal(0,0.0001,V))

            except ValueError as err:

                print err
                pdb.set_trace()
                # if hessian becomes negative definite, 
                # or any parameter becomes Inf or Nan during optimization,
                # re-optimize with a cold start
                x_init = x_init*np.exp(np.random.normal(0,0.0001,V))

        if self.restrict:
            self.seqparam['start'][1] = x_final[0]
        else:
            self.seqparam['start'][1:] = x_final[1:]
            self.seqparam['kozak'] = x_final[0]

    def __reduce__(self):
        return (rebuild_Transition, (self.seqparam, self.start, self.stop))

def rebuild_Transition(seqparam):
    t = Transition()
    t.seqparam = seqparam
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
    if restrict:
        xex[1] = x[0]
        xex[2:] = utils.MIN
    else:
        xex[1:] = x[1:]

    V = x.size
    func = 0
    df = np.zeros((V,), dtype=float)
    for datum,state,frame in zip(data,states,frames):

        for j from 0 <= j < 3:

            if restrict:
                arg = xex[datum.codon_id['start'][1:,j]]
            else:
                arg = x[0]*datum.codon_id['kozak'][1:,j] + xex[datum.codon_id['start'][1:,j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0] * arg \
                - state.pos_cross_moment_start[j].sum(1)[1:] * utils.nplog(1+np.exp(arg)))

            # evaluate gradient
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:] * logistic(-arg)
            if restrict:
                tmp = datum.codon_id['start'][1:,j]==1
                df[0] += frame.posterior[j] * np.sum(vec[tmp])
            else:
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
    if restrict:
        xex[1] = x[0]
        xex[2:] = utils.MIN
    else:
        xex[1:] = x[1:]

    V = x.size
    func = 0
    df = np.zeros((V,), dtype=float)
    Hf = np.zeros((V,V), dtype=float)

    for datum,state,frame in zip(data,states,frames):

        for j from 0 <= j < 3:

            if restrict:
                arg = xex[datum.codon_id['start'][1:,j]]
            else:
                arg = x[0]*datum.codon_id['kozak'][1:,j] + xex[datum.codon_id['start'][1:,j]]

            # evaluate function
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0] * arg \
                - state.pos_cross_moment_start[j].sum(1)[1:] * utils.nplog(1+np.exp(arg)))

            # evaluate gradient and hessian
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:] * logistic(-arg)
            vec2 = state.pos_cross_moment_start[j].sum(1)[1:] * logistic(arg) * logistic(-arg)
            if restrict:
                tmp = datum.codon_id['start'][1:,j]==1
                df[0] += frame.posterior[j] * np.sum(vec[tmp])
                Hf[0,0] += frame.posterior[j] * np.sum(vec2[tmp])
            else:
                df[0] += frame.posterior[j] * np.sum(vec*datum.codon_id['kozak'][1:,j])
                Hf[0,0] += frame.posterior[j] * np.sum(vec2*datum.codon_id['kozak'][1:,j]**2)
                for v from 1 <= v < V:
                    tmp = datum.codon_id['start'][1:,j]==v
                    df[v] += frame.posterior[j] * np.sum(vec[tmp])
                    Hf[v,v] += frame.posterior[j] * np.sum(vec2[tmp])
                    Hf[0,v] += frame.posterior[j] * np.sum(vec2[tmp] * datum.codon_id['kozak'][1:,j][tmp])

    if not restrict:
        Hf[:,0] = Hf[0,:]
    return -1.*func, -1.*df, Hf

cdef class Emission:

    def __cinit__(self):

        cdef long r
        cdef np.ndarray[np.float64_t, ndim=1] alpha_pattern
        cdef np.ndarray[np.float64_t, ndim=2] periodicity

        self.restrict = True
        self.S = 9
        self.R = len(utils.READ_LENGTHS)
        self.periodicity = np.empty((self.R,3,self.S), dtype=np.float64)
        self.logperiodicity = np.empty((self.R,3,self.S), dtype=np.float64)
        self.rescale = np.empty((self.R,self.S,8), dtype=np.float64)
        self.rate_alpha = np.empty((self.R,self.S), dtype=np.float64)
        self.rate_beta = np.empty((self.R,self.S), dtype=np.float64)
        alpha_pattern = np.array([1.,1.5,4.,6.,3.,6.,4.,1.,0.05])*100

        for r from 0 <= r < self.R:

            periodicity = np.ones((3,self.S), dtype=np.float64)
            periodicity[:,1:self.S-1] = np.random.rand(self.S-2)
            self.periodicity[r] = periodicity/periodicity.sum(0)
            self.logperiodicity[r] = np.log(self.periodicity[r])

            self.rate_alpha[r] = alpha_pattern*np.exp(np.random.normal(0,0.01,self.S))
            self.rate_beta[r] = 1.e4*np.random.rand(self.S)

        self.compute_rescaling()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_periodicity(self, data, states, frames):

        cdef bool optimized
        cdef long r, s, t, f, m, T
        cdef double ab
        cdef list constants
        cdef np.ndarray At, Bt, Ct, Et, result
        cdef Data datum
        cdef State state
        cdef Frame frame

        T = len(data)
        Et = np.array([d.scale for d in data]).reshape(T,1)

        for r from 0 <= r < self.R:

            for s from 1 <= s < self.S-1:
            
                # compute constants
                At = np.zeros((3,), dtype=np.float64)
                Bt = np.zeros((T,3), dtype=np.float64)
                Ct = np.zeros((T,3), dtype=np.float64)
                ab = self.alpha[r,s]*self.beta[r,s]
                for t,(datum,state,frame) in enumerate(zip(data,states,frames)):
                    
                    for f in xrange(3):
                        
                        At[0] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*datum.obs[3*m+f,r] 
                              for m in xrange(datum.M) if datum.mappable[3*m+f] and 
                              (datum.mappable[3*m+f+1] or datum.mappable[3*m+f+2])])
                        At[1] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*datum.obs[3*m+f+1,r] 
                              for m in xrange(datum.M) if datum.mappable[3*m+f+1] and 
                              (datum.mappable[3*m+f] or datum.mappable[3*m+f+2])])
                        At[2] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*datum.obs[3*m+f+2,r] 
                              for m in xrange(datum.M) if datum.mappable[3*m+f+2] and 
                              (datum.mappable[3*m+f] or datum.mappable[3*m+f+1])])
                        Bt[t,0] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if not datum.mappable[3*m+f] 
                                   and datum.mappable[3*m+f+1] and datum.mappable[3*m+f+2]])
                        Bt[t,1] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if datum.mappable[3*m+f] 
                                   and not datum.mappable[3*m+f+1] and datum.mappable[3*m+f+2]])
                        Bt[t,2] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if datum.mappable[3*m+f] 
                                   and datum.mappable[3*m+f+1] and not datum.mappable[3*m+f+2]])
                        Ct[t,0] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if datum.mappable[3*m+f] 
                                   and not datum.mappable[3*m+f+1] and not datum.mappable[3*m+f+2]])
                        Ct[t,1] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if not datum.mappable[3*m+f] 
                                   and datum.mappable[3*m+f+1] and not datum.mappable[3*m+f+2]])
                        Ct[t,2] += frame.max_posterior[f] * np.array([state.pos_first_moment[f,m,s]*(datum.total[f,m,r]+ab) 
                                   for m in xrange(self.M) if not datum.mappable[3*m+f] 
                                   and not datum.mappable[3*m+f+1] and datum.mappable[3*m+f+2]])

                constants = [At, Bt, Ct, Et, self.beta[r,s]]

                # run optimizer
                result, optimized = optimize_periodicity(self.periodicity[r,:,s], constants)
                if optimized:
                    self.periodicity[r,:,s] = result

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
                
                    self.rescale[r,s,j] = np.sum(self.periodicity[r,mask,s])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_beta(self, data, states, frames):

        cdef bool optimized
        cdef long r, s
        cdef list constants
        cdef np.ndarray x_init, x_final

        for r in xrange(data[0].R):

            for s in xrange(states[0].S):

                # warm start for the optimization
                optimized = False
                x_init = np.array([[self.rate_beta[r,s]]])
                constants = [r, s, self.rate_alpha[r,s]]

                while not optimized:

                    try:
                        x_final, optimized = optimize_beta(x_init, data, states, frames, self.rescale, constants)
                        if optimized:
                            self.rate_beta[r,s] = x_final[0,0]

                    except ValueError:
                        # if any parameter becomes Inf or Nan during optimization,
                        # or if hessian is negative definite, re-optimize with a cold start
                        x_init = x_init*(1+0.1*(np.random.rand(1,1)-0.5))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_alpha(self, data, states, frames):

        cdef bool optimized
        cdef long r, s
        cdef list constants
        cdef np.ndarray x_init, x_final

        for r in xrange(data[0].R):

            for s in xrange(states[0].S):

                # warm start for the optimization
                optimized = False
                x_init = np.array([[self.rate_beta[r,s]]])
                constants = [r, s, self.rate_alpha[r,s]]

                while not optimized:

                    try:
                        x_final, optimized = optimize_alpha(x_init, data, states, frames, self.rescale, constants)
                        if optimized:
                            self.rate_alpha[r,s] = x_final[0,0]

                    except ValueError:
                        # if any parameter becomes Inf or Nan during optimization,
                        # or if hessian is negative definite, re-optimize with a cold start
                        x_init = x_init*(1+0.1*(np.random.rand(1,1)-0.5))

    def __reduce__(self):
        return (rebuild_Emission, (self.periodicity, self.rate_alpha, self.rate_beta))

def rebuild_Emission(periodicity, alpha, beta):
    e = Emission()
    e.periodicity = periodicity
    e.logperiodicity = np.log(periodicity)
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
        func = np.sum(At * np.log(xx)) - 
               np.sum(Bt * np.log((1-xx)*Et+ab),0) -
               np.sum(Ct * np.log(xx*Et+ab),0)
        if np.isnan(func) or np.isinf(func):
            f = np.array([np.finfo(np.float32).max]).astype(np.float64)
        else:
            f = np.array([-1*func]).astype(np.float64)

        # compute gradient
        Df = At/xx[0] + np.sum(Bt * Et/((1-xx)*Et+ab),0) - 
             np.sum(Ct * Et/(xx*Et+ab),0)
        if np.isnan(Df).any() or np.isinf(Df).any():
            Df = -1 * np.finfo(np.float32).max * 
                 np.ones((1,xx.size), dtype=np.float64)
        else:
            Df = -1*Df.reshape(1,xx.size)

        if z is None:
            return cvx.matrix(f), cvx.matrix(Df)

        # compute hessian
        hess = 1.*At/xx[0]**2 - np.sum(Bt * Et**2/((1-xx)*Et+ab)**2,0) -
               np.sum(Ct * Et**2/(xx*Et+ab)**2,0)

        # check if hessian is positive semi-definite
        if np.any(hess<0):
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

def optimize_beta(x_init, data, states, frames, rescale, constants):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:
            # compute likelihood function and gradient
            results = beta_func_grad(xx, data, states, frames, rescale, constants)

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
            results = beta_func_grad_hess(xx, data, states, frames, rescale, constants)

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
            if hess[0,0]<0:
                raise ValueError
            hess = z[0] * hess

            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((1,), dtype='float')))
    h = cvx.matrix(np.zeros((1,1), dtype='float'))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple beta_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, list constants):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, s, f, l
    cdef double alpha, func
    cdef np.ndarray gradient, mask, new_scale

    r = constants[0]
    s = constants[1]
    alpha = constants[2]

    func = 0
    gradient = np.zeros((1,), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):
                    
        for f in xrange(3):

            new_scale = rescale[r,s,datum.rescale_indices[r,f,:]]
            mask = new_scale!=0
            func = func + frame.max_posterior * np.sum(state.pos_first_moment[f,mask,s] * 
                   (alpha*x*np.log(x) + gammaln(datum.total[f,mask,r]+alpha*x) - 
                    gammaln(alpha*x) - (datum.total[f,mask,r]+alpha*x)*np.log(datum.scale*new_scale+x)))

            gradient = gradient + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                       (alpha * (np.log(x) + 1 + digamma(datum.total[f,mask,r]+alpha*x) - 
                        digamma(alpha*x) - np.log(datum.scale*new_scale+x)) + 
                       (datum.total[f,mask,r]+alpha*x)/(datum.scale*new_scale+x)))

            if s==0:

                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (alpha*x*np.log(x) + 
                               gammaln(datum.obs[l,r]+alpha*x) - gammaln(alpha*x) - 
                               (datum.obs[l,r]+alpha*x)*np.log(datum.scale/3.+x))
                        gradient = gradient + frame.posterior[f] * (alpha * (np.log(x) + 
                                   1 + digamma(datum.obs[l,r]+alpha*x) - digamma(alpha*x) - 
                                   np.log(datum.scale/3.+x)) + (datum.obs[l,r]+alpha*x)/(datum.scale/3.+x))

            if s==8:

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (alpha*x*np.log(x) + 
                               gammaln(datum.obs[l,r]+alpha*x) - gammaln(alpha*x) - 
                               (datum.obs[l,r]+alpha*x)*np.log(datum.scale/3.+x))
                        gradient = gradient + frame.posterior[f] * (alpha * (np.log(x) + 
                                   1 + digamma(datum.obs[l,r]+alpha*x) - digamma(alpha*x) - 
                                   np.log(datum.scale/3.+x)) + (datum.obs[l,r]+alpha*x)/(datum.scale/3.+x))

    func = -1.*func
    gradient = -1.*gradient

    return func, gradient

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple beta_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, list constants):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, s, f, l
    cdef double alpha, func
    cdef np.ndarray gradient, hessian, new_scale, mask

    r = constants[0]
    s = constants[1]
    alpha = constants[2]

    func = 0
    gradient = np.zeros((1,), dtype=np.float64)
    hessian = np.zeros((1,1), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):
                    
        for f in xrange(3):

            new_scale = rescale[r,s,datum.rescale_indices[r,f,:]]
            mask = new_scale!=0
            func = func + frame.max_posterior * np.sum(state.pos_first_moment[f,mask,s] * 
                   (alpha*x*np.log(x) + gammaln(datum.total[f,mask,r]+alpha*x) - 
                    gammaln(alpha*x) - (datum.total[f,mask,r]+alpha*x)*np.log(datum.scale*new_scale+x)))

            gradient = gradient + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                       (alpha * (np.log(x) + 1 + digamma(datum.total[f,mask,r]+alpha*x) - 
                        digamma(alpha*x) - np.log(datum.scale*new_scale+x)) + 
                       (datum.total[f,mask,r]+alpha*x)/(datum.scale*new_scale+x)))

            hessian = hessian + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                       (alpha * (alpha*polygamma(1,datum.total[f,mask,r]+alpha*x) - 
                        alpha*polygamma(1,alpha*x) + (datum.scale*new_scale+1)/(datum.scale*new_scale+x)) - 
                       (datum.total[f,mask,r]+alpha*x)/(datum.scale*new_scale+x)**2))

            if s==0:

                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (alpha*x*np.log(x) + 
                               gammaln(datum.obs[l,r]+alpha*x) - gammaln(alpha*x) - 
                               (datum.obs[l,r]+alpha*x)*np.log(datum.scale/3.+x))
                        gradient = gradient + frame.posterior[f] * (alpha * (np.log(x) + 
                                   1 + digamma(datum.obs[l,r]+alpha*x) - digamma(alpha*x) - 
                                   np.log(datum.scale/3.+x)) + (datum.obs[l,r]+alpha*x)/(datum.scale/3.+x))
                        hessian = hessian + frame.posterior[f] * (alpha * 
                                  (alpha*polygamma(1,datum.obs[l,r]+alpha*x) - alpha*polygamma(1,alpha*x) + 
                                  (datum.scale/3.+1)/(datum.scale/3.+x)) - (datum.total[l,r]+alpha*x)/(datum.scale/3.+x)**2)

            if s==8:

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (alpha*x*np.log(x) + 
                               gammaln(datum.obs[l,r]+alpha*x) - gammaln(alpha*x) - 
                               (datum.obs[l,r]+alpha*x)*np.log(datum.scale/3.+x))
                        gradient = gradient + frame.posterior[f] * (alpha * (np.log(x) + 
                                   1 + digamma(datum.obs[l,r]+alpha*x) - digamma(alpha*x) - 
                                   np.log(datum.scale/3.+x)) + (datum.obs[l,r]+alpha*x)/(datum.scale/3.+x))
                        hessian = hessian + frame.posterior[f] * (alpha * 
                                  (alpha*polygamma(1,datum.obs[l,r]+alpha*x) - alpha*polygamma(1,alpha*x) + 
                                  (datum.scale/3.+1)/(datum.scale/3.+x)) - (datum.total[l,r]+alpha*x)/(datum.scale/3.+x)**2)

    func = -1.*func
    gradient = -1.*gradient
    hessian = -1.*hessian

    return func, gradient, hessian

def optimize_alpha(x_init, data, states, frames, rescale, constants):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:
            # compute likelihood function and gradient
            results = alpha_func_grad(xx, data, states, frames, rescale, constants)

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
            results = alpha_func_grad_hess(xx, data, states, frames, rescale, constants)

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
            if hess[0,0]<0:
                raise ValueError
            hess = z[0] * hess

            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((1,), dtype=np.float64)))
    h = cvx.matrix(np.zeros((1,1), dtype=np.float64))

    # call a constrained nonlinear solver
    solution = solvers.cp(F, G=G, h=h)

    if solution['status'] in ['optimal','unknown']:
        optimized = True
    else:
        optimized = False
    x_final = np.array(solution['x']).ravel()

    return x_final, optimized

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple alpha_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, list constants):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, s, f, l
    cdef double beta, func
    cdef np.ndarray gradient, mask, new_scale

    r = constants[0]
    s = constants[1]
    beta = constants[2]

    func = 0
    gradient = np.zeros((1,), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):
                    
        for f in xrange(3):

            new_scale = rescale[r,s,datum.rescale_indices[r,f,:]]
            mask = new_scale!=0
            func = func + frame.max_posterior * np.sum(state.pos_first_moment[f,mask,s] * 
                   (x*beta*np.log(beta) + gammaln(datum.total[f,mask,r]+x*beta) - 
                    gammaln(x*beta) - (datum.total[f,mask,r]+x*beta)*np.log(datum.scale*new_scale+beta)))

            gradient = gradient + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                       (beta * (np.log(beta) + digamma(datum.total[f,mask,r]+x*beta) - 
                        digamma(x*beta) - np.log(datum.scale*new_scale+beta))))

            if s==0:

                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x*beta*np.log(beta) + 
                               gammaln(datum.obs[l,r]+x*beta) - gammaln(x*beta) - 
                               (datum.obs[l,r]+x*beta) * np.log(datum.scale/3.+beta))
                        gradient = gradient + frame.posterior[f] * beta * (np.log(beta) + 
                                   digamma(datum.obs[l,r]+x*beta) - digamma(x*beta) - 
                                   np.log(datum.scale/3.+beta))

            if s==8:

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x*beta*np.log(beta) + 
                               gammaln(datum.obs[l,r]+x*beta) - gammaln(x*beta) - 
                               (datum.obs[l,r]+x*beta) * np.log(datum.scale/3.+beta))
                        gradient = gradient + frame.posterior[f] * beta * (np.log(beta) + 
                                   digamma(datum.obs[l,r]+x*beta) - digamma(x*beta) - 
                                   np.log(datum.scale/3.+beta))

    func = -1.*func
    gradient = -1.*gradient

    return func, gradient

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple alpha_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, list states, list frames, np.ndarray[np.float64_t, ndim=3] rescale, list constants):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, s, f
    cdef double beta, func
    cdef np.ndarray gradient, mask, new_scale

    r = constants[0]
    s = constants[1]
    beta = constants[2]

    func = 0
    gradient = np.zeros((1,), dtype=np.float64)
    hessian = np.zeros((1,1), dtype=np.float64)
    for datum,state,frame in zip(data,states,frames):
                    
        for f in xrange(3):

            new_scale = rescale[r,s,datum.rescale_indices[r,f,:]]
            mask = new_scale!=0
            func = func + frame.max_posterior * np.sum(state.pos_first_moment[f,mask,s] * 
                   (x*beta*np.log(beta) + gammaln(datum.total[f,mask,r]+x*beta) - 
                    gammaln(x*beta) - (datum.total[f,mask,r]+x*beta)*np.log(datum.scale*new_scale+beta)))

            gradient = gradient + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                       (beta * (np.log(beta) + digamma(datum.total[f,mask,r]+x*beta) - 
                        digamma(x*beta) - np.log(datum.scale*new_scale+beta))))

            hessian = hessian + frame.max_posterior[f] * np.sum(state.pos_first_moment[f,mask,s] * 
                      beta**2 * (polygamma(1,datum.total[f,mask,r]+x*beta) - 
                      polygamma(1,x*beta)))

            if s==0:

                # add extra terms for first state
                for l from 0 <= l < f:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x*beta*np.log(beta) + 
                               gammaln(datum.obs[l,r]+x*beta) - gammaln(x*beta) - 
                               (datum.obs[l,r]+x*beta) * np.log(datum.scale/3.+beta))
                        gradient = gradient + frame.posterior[f] * beta * (np.log(beta) + 
                                   digamma(datum.obs[l,r]+x*beta) - digamma(x*beta) - 
                                   np.log(datum.scale/3.+beta))
                        hessian = hessian + frame.posterior[f] * beta**2 * 
                                  (polygamma(1,datum.obs[l,r]+x*beta) - polygamma(1,x*beta))

            if s==8:

                # add extra terms for last state
                for l from 3*datum.M+f <= l < datum.L:
                    if datum.mappable[l,r]:
                        func = func + frame.posterior[f] * (x*beta*np.log(beta) + 
                               gammaln(datum.obs[l,r]+x*beta) - gammaln(x*beta) - 
                               (datum.obs[l,r]+x*beta) * np.log(datum.scale/3.+beta))
                        gradient = gradient + frame.posterior[f] * beta * (np.log(beta) + 
                                   digamma(datum.obs[l,r]+x*beta) - digamma(x*beta) - 
                                   np.log(datum.scale/3.+beta))
                        hessian = hessian + frame.posterior[f] * beta**2 * 
                                  (polygamma(1,datum.obs[l,r]+x*beta) - polygamma(1,x*beta))

    func = -1.*func
    gradient = -1.*gradient
    hessian = -1.*hessian

    return func, gradient, hessian

def learn_parameters(observations, codon_id, scales, mappability, restarts, mintol):

    cdef long restart, i
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
    Ls = []
    Lmax = -np.inf

    for restart in xrange(restarts):

        # Initialize latent variables
        states = [State(datum.M) for datum in data]
        frames = [Frame() for datum in data]

        # First, allow start transitions only at AUG
        print "Estimating periodicity using canonical START codon ..."
        transition = Transition()
        transition.seqparam['start'][2:] = utils.MIN
        emission = Emission()

        starttime = time.time()
        # compute initial log likelihood
        for datum,state in zip(data,states):
            datum.compute_log_likelihood(emission)
            state._forward_update(datum, transition)
        L = np.mean([(np.sum(frame.posterior*state.likelihood) \
                + np.sum(frame.posterior*datum.extra_log_likelihood))/datum.L \
                for datum,state,frame in zip(data,states,frames)])
        dL = np.inf
        print "%.2f sec to compute likelihood"%(time.time()-starttime)

        # iterate gradient descent till weak convergence
        # occupancy parameters are not optimized in this initial stage
        reltol = np.abs(dL)/np.abs(L)
        while reltol>mintol*1e2:

            totaltime = time.time()

            # update latent states
            starttime = time.time()
            for state,frame,datum in zip(states, frames, data):
                state._reverse_update(datum, transition)
                frame.update(datum, state)
            print "%.2f sec to update latent states"%(time.time()-starttime)

            # update periodicity parameters
            starttime = time.time()
            emission.update_periodicity(data, states, frames)
            print "%.2f sec to update emission"%(time.time()-starttime)

            # update transition parameters
            starttime = time.time()
            transition.update(data, states, frames)
            print "%.2f sec to update transition"%(time.time()-starttime)

            # compute log likelihood
            starttime = time.time()
            for datum,state in zip(data,states):
                datum.compute_log_likelihood(emission)
                state._forward_update(datum, transition)
            newL = np.mean([(np.sum(frame.posterior*state.likelihood) \
                + np.sum(frame.posterior*datum.extra_log_likelihood))/datum.L \
                for datum,state,frame in zip(data,states,frames)])
            print "%.2f sec to compute likelihood"%(time.time()-starttime)

            dL = newL-L
            L = newL
            reltol = dL/np.abs(L)
            print L, reltol, time.time()-totaltime

        # update emission occupancy parameters, keeping all other parameters fixed
        print "Keeping transition parameters fixed, update emission occupancy parameters ..."
        emission.restrict = False

        # update occupancy precision
        starttime = time.time()
        emission.update_beta(data, states, frames)
        print "%.2f sec to update beta"%(time.time()-starttime)

        # update occupancy mean
        starttime = time.time()
        emission.update_alpha(data, states, frames)
        print "%.2f sec to update alpha"%(time.time()-starttime)

        # Next, keeping emission parameters fixed,
        # relax transitions to allow noncanonical codons
        print "Keeping emission parameters fixed, estimating transition probabilities ..."
        transition.restrict = False
        starttime = time.time()
        transition.seqparam['start'][2:] = -5+np.random.rand(transition.seqparam['start'][2:].size)

        # update likelihood with updated emission parameters
        # update posterior of latent states with initialization of transition parameters
        for datum,state,frame in zip(data,states,frames):
            datum.compute_log_likelihood(emission)
            state._forward_update(datum, transition)
            state._reverse_update(datum, transition)
            frame.update(datum, state)
        transition.update(data, states, frames)
        print "%.2f sec to update transition"%(time.time()-starttime)

        # compute log likelihood
        starttime = time.time()
        for datum,state in zip(data,states):
            datum.compute_log_likelihood(emission)
            state._forward_update(datum, transition)
        L = np.mean([(np.sum(frame.posterior*state.likelihood) \
                + np.sum(frame.posterior*datum.extra_log_likelihood))/datum.L \
                for datum,state,frame in zip(data,states,frames)])
        print "%.2f sec to compute likelihood"%(time.time()-starttime)

        # Finally, update all parameters
        print "Updating all parameters"
        dL = np.inf
        reltol = np.abs(dL)/np.abs(L)
        while reltol>mintol:

            totaltime = time.time()

            # update latent variables
            starttime = time.time()
            for state,frame,datum in zip(states, frames, data):
                state._reverse_update(datum, transition)
                frame.update(datum, state)
            print "%.2f sec to update latent states"%(time.time()-starttime)

            # update transition parameters
            starttime = time.time()
            transition.update(data, states, frames)
            print "%.2f sec to update transition"%(time.time()-starttime)

            # update emission parameters
            starttime = time.time()
            emission.update_periodicity(data, states, frames)
            emission.update_beta(data, states, frames)
            emission.update_alpha(data, states, frames)
            print "%.2f sec to update emission"%(time.time()-starttime)

            # compute log likelihood
            starttime = time.time()
            for datum,state in zip(data, states):
                datum.compute_log_likelihood(emission)
                state._forward_update(datum, transition)
            newL = np.mean([(np.sum(frame.posterior*state.likelihood) \
                + np.sum(frame.posterior*datum.extra_log_likelihood))/datum.L \
                for datum,state,frame in zip(data,states,frames)])
            print "%.2f sec to compute likelihood"%(time.time()-starttime)

            dL = newL-L
            L = newL
            reltol = dL/np.abs(L)
            print L, reltol, time.time()-totaltime

        Ls.append(L)
        if L>Lmax:
            Lmax = L
            best_transition = transition
            best_emission = emission

    return best_transition, best_emission, Ls

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
        try:
            datum.compute_log_likelihood(emission)
            state._forward_update(datum, transition)
            frame.update(datum, state)
            state.decode(datum, transition, emission, frame)
        except ValueError:
            pass

    return states, frames

