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

    def __cinit__(self, np.ndarray[np.uint64_t, ndim=2] obs, \
    dict codon_id, double scale, \
    np.ndarray[np.uint8_t, ndim=2, cast=True] missing):

        cdef double r,m,f

        self.L = obs.shape[0]
        self.M = self.L/3-1
        self.R = obs.shape[1]
        self.obs = obs
        self.scale = scale
        self.missing = missing
        self.codon_id = codon_id
        self.indices = [[[np.empty((1,), dtype=np.uint64) \
            for s in xrange(9)] for r in xrange(4)] for f in xrange(3)]
        self.total = np.empty((3,self.M,self.R), dtype=np.uint64)
        for f from 0 <= f < 3:
            for r from 0 <= r < self.R:
                self.total[f,:,r] = np.array([self.obs[3*m+f:3*m+3+f,r][self.missing[3*m+f:3*m+3+f,r]].sum() \
                    for m from 0 <= m < self.M])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef compute_log_likelihood(self, Emission emission):

        cdef long r, f, m, l
        cdef np.ndarray[np.float64_t, ndim=2] log_likelihood, rate_log_likelihood
        cdef np.ndarray[np.uint64_t, ndim=2] count_data
        cdef np.ndarray[np.int64_t, ndim=1] missing

        self.log_likelihood = np.zeros((3, self.M, emission.S), dtype=np.float64)
        self.extra_log_likelihood = np.zeros((3,), dtype=np.float64)

        for f from 0 <= f < 3:

            for r from 0 <= r < self.R:

                log_likelihood = np.zeros((self.M,emission.S), dtype=np.float64)

                # periodicity likelihood, accounting for mappability
                log_likelihood = gammaln(self.total[f,:,r:r+1]+1) - utils.insum(gammaln(count_data+1),[1])
                log_likelihood += np.array([0 if np.all(self.missing[3*m+f:3*m+3+f,r]) else \
                    self.obs[3*m+f:3*m+3+f,r][self.missing[3*m+f:3*m+3+f,r]] \
                    * (emission.logperiodicity[r,self.missing[3*m+f:3*m+3+f,r],:] \
                    - np.log(utils.insum(emission.periodicity[r,self.missing[3*m+f:3*m+3+f,r],:],[1]))) \
                    if np.any(self.missing[3*m+f:3*m+3+f,r]) else \
                    np.sum(self.obs[3*m+f:3*m+3+f,r]*emission.logperiodicity[r],1) \
                    for m in xrange(self.M)])

                if not emission.restrict:

                    # occupancy likelihood, accounting for mappability
                    scale = self.scale * np.array([np.zeros(emission.S) if np.all(self.missing[3*m+f:3*m+3+f,r]) \
                        else emission.periodicity[r,self.missing[3*m+f:3*m+3+f,r],:].sum(0) for m in xrange(self.M)])
                    missing = np.where(scale==0)[0].astype(np.int64)
                    scale[missing] = 1e-8
                    rate_log_likelihood = emission.rate_alpha[r]*emission.rate_beta[r]*utils.nplog(emission.rate_beta[r]) \
                        - (emission.rate_alpha[r]*emission.rate_beta[r]+self.total[f,:,r:r+1])*utils.nplog(emission.rate_beta[r]+scale) \
                        + gammaln(emission.rate_alpha[r]*emission.rate_beta[r]+self.total[f,:,r:r+1]) \
                        - gammaln(emission.rate_alpha[r]*emission.rate_beta[r]) \
                        + self.total[f,:,r:r+1]*utils.nplog(scale) - gammaln(self.total[f,:,r:r+1]+1)
                    rate_log_likelihood[missing] = 0
                    log_likelihood += rate_log_likelihood

                    # likelihood of extra positions
                    for l from 0 <= l < f:
                        if not self.missing[l,r]:
                            self.extra_log_likelihood[f] += emission.rate_alpha[r,0]*emission.rate_beta[r,0]*utils.nplog(emission.rate_beta[r,0]) \
                                - (emission.rate_alpha[r,0]*emission.rate_beta[r,0]+self.obs[l,r])*utils.nplog(emission.rate_beta[r,0]+self.scale/3.) \
                                + gammaln(emission.rate_alpha[r,0]*emission.rate_beta[r,0]+self.obs[l,r]) \
                                - gammaln(emission.rate_alpha[r,0]*emission.rate_beta[r,0]) \
                                + self.obs[l,r]*utils.nplog(self.scale/3.) - gammaln(self.obs[l,r]+1)
                    for l from 3*self.M+f <= l < self.L:
                        if not self.missing[l,r]:
                            self.extra_log_likelihood[f] += emission.rate_alpha[r,emission.S-1]*emission.rate_beta[r,emission.S-1]*utils.nplog(emission.rate_beta[r,emission.S-1]) \
                                    - (emission.rate_alpha[r,emission.S-1]*emission.rate_beta[r,emission.S-1]+self.obs[l,r])*utils.nplog(emission.rate_beta[r,emission.S-1]+self.scale/3.) \
                                    + gammaln(emission.rate_alpha[r,emission.S-1]*emission.rate_beta[r,emission.S-1]+self.obs[l,r]) \
                                    - gammaln(emission.rate_alpha[r,emission.S-1]*emission.rate_beta[r,emission.S-1]) \
                                    + self.obs[l,r]*utils.nplog(self.scale/3.) - gammaln(self.obs[l,r]+1)

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

        self.M = M
        self.S = 9
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

        if transition.start=='canonical':
            P = logistic(-1*transition.seqparam['start'][data.codon_id['start']])
        else:
            P = logistic(-1*(transition.seqparam['kozak']*data.codon_id['kozak']+transition.seqparam['start'][data.codon_id['start']]))
        Q = logistic(-1*transition.seqparam['stop'][data.codon_id['stop']])

        for f from 0 <= f < 3:

            newalpha = logprior + data.log_likelihood[f,0,:]
            L = normalize(newalpha)
            for s from 0 <= s < self.S:
                self.alpha[f,0,s] = newalpha[s] - L
            self.likelihood[0,f] = L

            for m from 1 <= m < self.M:
      
                # other states
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
        if transition.stop=='readthrough':
            self.pos_cross_moment_stop = np.zeros((3,4,2), dtype=np.float64)

        if transition.start=='canonical':
            P = logistic(-1*transition.seqparam['start'][data.codon_id['start']])
        else:
            P = logistic(-1*(transition.seqparam['kozak']*data.codon_id['kozak']+transition.seqparam['start'][data.codon_id['start']]))
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
    
                # pos cross moment at stop
                if transition.stop=='readthrough':
                    if data.codon_id['stop'][m+1,f]!=0:
                        a = self.alpha[f,m,4] - self.likelihood[m+1,f]
                        self.pos_cross_moment_stop[f,data.codon_id['stop'][m+1,f],0] += exp(a+q)
                        self.pos_cross_moment_stop[f,data.codon_id['stop'][m+1,f],1] += exp(a+qq)

                # other states
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

        if transition.stop=='readthrough':
            if np.isnan(self.pos_cross_moment_stop).any() \
            or np.isinf(self.pos_cross_moment_stop).any():
                print "Warning: Inf/Nan in stop cross moment"
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

        P = logistic(-1*(transition.seqparam['kozak']*data.codon_id['kozak']+transition.seqparam['start'][data.codon_id['start']]))
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

                # other states
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

            state[self.M-1] = np.argmax(alpha)
            for m in xrange(self.M-2,0,-1):
                state[m] = pointer[m+1,state[m+1]]
            state[0] = pointer[0,0]
            try:
                self.best_start.append(np.where(state==2)[0][0]*3+f)
            except IndexError:
                self.best_start.append(None)
            try:
                self.best_stop.append(np.where(state==7)[0][0]*3+f)
            except IndexError:
                self.best_stop.append(None)
            self.max_posterior[f] = exp(np.max(alpha) - np.sum(self.likelihood[:,f]))

        self.alpha = np.empty((1,1,1), dtype=np.float64)
        self.pos_cross_moment_start = np.empty((1,1,1), dtype=np.float64)
        self.pos_cross_moment_stop = np.empty((1,1,1), dtype=np.float64)
        self.pos_first_moment = np.empty((1,1,1), dtype=np.float64)
        self.likelihood = np.empty((1,1), dtype=np.float64)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double joint_likelihood(self, Data data, Transition transition, np.ndarray[np.uint8_t, ndim=1] state, long frame):

        cdef long m
        cdef double p, q, joint_likelihood

        joint_likelihood = data.log_likelihood[frame,0,state[0]]

        for m from 1 <= m < self.M:

            if state[m-1]==0:
    
                p = transition.seqparam['kozak']*data.codon_id['kozak'][m,frame] \
                    + transition.seqparam['start'][data.codon_id['start'][m,frame]]
                try:
                    joint_likelihood = joint_likelihood - log(1+exp(-p))
                    if state[m]==0:
                        joint_likelihood = joint_likelihood - p
                except OverflowError:
                    if state[m]==1:
                        joint_likelihood = joint_likelihood - p

            elif state[m-1]==4:

                q = transition.seqparam['stop'][data.codon_id['stop'][m,frame]]
                try:
                    joint_likelihood = joint_likelihood - log(1+exp(-q))
                    if state[m]==4:
                        joint_likelihood = joint_likelihood - q
                except OverflowError:
                    if state[m]==5:
                        joint_likelihood = joint_likelihood - q

            joint_likelihood = joint_likelihood + data.log_likelihood[frame,m,state[m]]

        return joint_likelihood

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double compute_posterior(self, Data data, Transition transition, long start, long stop):

        cdef long frame
        cdef double joint_like, like, posterior
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

        joint_like = self.joint_likelihood(data, transition, state, frame)
        like = np.sum(self.likelihood[:,frame])
        posterior = exp(joint_like - like)

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

        self.S = 9
        self.restrict = True
        self.C = len(set(utils.STARTS.values()))+1

        self.seqparam = dict()

        # initialize translation initiation parameters
        self.seqparam['kozak'] = 0
        self.seqparam['start'] = -1*np.random.rand(self.C)
        self.seqparam['start'][0] = utils.MIN
        self.seqparam['start'][1] = 1+np.random.rand()

        # initialize translation termination parameters
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
cdef tuple transition_func_grad(np.ndarray[np.float64_t, ndim=1] x, \
list data, list states, list frames, bool restrict):

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
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0]*arg \
                - state.pos_cross_moment_start[j].sum(1)[1:]*utils.nplog(1+np.exp(arg)))

            # evaluate gradient
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:]*logistic(-arg)
            if restrict:
                tmp = datum.codon_id['start'][1:,j]==1
                df[0] += frame.posterior[j] * np.sum(vec[tmp])
            else:
                df[0] += frame.posterior[j]*np.sum(vec*datum.codon_id['kozak'][1:,j])
                for v from 1 <= v < V:
                    tmp = datum.codon_id['start'][1:,j]==v
                    df[v] += frame.posterior[j] * np.sum(vec[tmp])

    return -1.*func, -1.*df

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple transition_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, \
list data, list states, list frames, bool restrict):

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
            func += frame.posterior[j] * np.sum(state.pos_cross_moment_start[j,1:,0]*arg \
                - state.pos_cross_moment_start[j].sum(1)[1:]*utils.nplog(1+np.exp(arg)))

            # evaluate gradient and hessian
            vec = state.pos_cross_moment_start[j,1:,0] \
                - state.pos_cross_moment_start[j].sum(1)[1:]*logistic(-arg)
            vec2 = state.pos_cross_moment_start[j].sum(1)[1:]*logistic(arg)*logistic(-arg)
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
                    Hf[0,v] += frame.posterior[j] * np.sum(vec2[tmp]*datum.codon_id['kozak'][1:,j][tmp])

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update_periodicity(self, list data, list states, list frames):

        cdef long r, j, index, f, m
        cdef np.ndarray[np.float64_t, ndim=2] periodicity, count_data
        cdef np.ndarray[np.float64_t, ndim=1] period
        cdef np.ndarray[np.int64_t, ndim=1] notmissing
        cdef Data datum
        cdef State state
        cdef Frame frame

        for r from 0 <= r < self.R:

            periodicity = np.zeros((3,self.S), dtype=np.float64)
            for f from 0 <= f < 3:

                for datum,state,frame in zip(data,states,frames):

                    count_data = np.array([datum.obs[3*m+f:3*m+3+f,r] for m in xrange(datum.M) \
                        if not np.any(datum.missing[3*m+f:3*m+3+f,r])]).astype(np.float64)
                    notmissing = np.array([m for m in xrange(datum.M) \
                        if not np.any(datum.missing[3*m+f:3*m+3+f,r])])
                    periodicity += frame.posterior[f] \
                        * np.dot(count_data.T, state.pos_first_moment[f,notmissing,:])

            # shared parameters for CDS state across frames, and TSS+ state in frame 0
            period = periodicity[:,3] + periodicity[:,4]
            periodicity[:,3] = period
            periodicity[:,4] = period
            periodicity[:,0] = 1
            periodicity[:,self.S-1] = 1

            periodicity[periodicity==0] = 1e-10
            periodicity = periodicity/periodicity.sum(0)
            self.logperiodicity[r,:,:] = utils.nplog(periodicity)

        if np.isinf(self.logperiodicity).any() or np.isnan(self.logperiodicity).any():
            print "Warning: Inf/Nan in periodicity parameter"

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef update_beta(self, list data, list states, list frames, double reltol):

        cdef long m, r, f, l, iter, maxiters
        cdef double a, relerr
        cdef np.ndarray[np.int64_t, ndim=1] nmissing
        cdef np.ndarray[np.float64_t, ndim=1] tmpa, tmpb
        cdef np.ndarray[np.float64_t, ndim=2] bb, den, mden, eden, numbb, newbb, tmpc
        cdef Data datum
        cdef State state
        cdef Frame frame
        cdef list notmissing, nmiss, main_den, extra_den

        maxiters = 1000
        bb = self.rate_beta
        den = np.zeros((4,self.S), dtype=float)
        notmissing = []
        main_den = []
        extra_den = []

        for datum,state,frame in zip(data,states,frames):
            nmiss = []
            mden = np.zeros((4,self.S), dtype=float)
            eden = np.zeros((4,self.S), dtype=float)
            for r from 0 <= r < 4:
                nmiss.append([])
                for f from 0 <= f < 3:
                    nmissing = np.array([m for m in xrange(datum.M) \
                        if not np.any(datum.missing[3*m+f:3*m+3+f,r])])
                    mden[r] += frame.posterior[f]*np.sum(state.pos_first_moment[f,nmissing,:],0)
                    for l from 0 <= l < f:
                        if not datum.missing[l,r]:
                            eden[r,0] = eden[r,0] + frame.posterior[f]
                    for l from 3*datum.M+f <= l < datum.L:
                        if not datum.missing[l,r]:
                            eden[r,8] = eden[r,8] + frame.posterior[f]
                    nmiss[r].append(nmissing)
            main_den.append(mden)
            extra_den.append(eden)
            den = den + mden + eden
            notmissing.append(nmiss)

        relerr = np.inf
        for iter from 0 <= iter < maxiters:

            numbb = np.zeros((4,self.S), dtype=float)

            for datum,state,frame,mden,eden,nmiss in zip(data,states,frames,main_den,extra_den,notmissing):

                numbb += np.log(bb+datum.scale)*mden + np.log(bb+datum.scale/3.)*eden
                for r from 0 <= r < 4:

                    tmpa = np.zeros((self.S,), dtype='float')
                    tmpb = np.zeros((self.S,), dtype='float')
                    for f from 0 <= f < 3:

                        tmpc = self.rate_alpha[r]*bb[r] + datum.total[f,nmiss[r][f],r:r+1]
                        tmpa = tmpa + frame.posterior[f] * np.sum(state.pos_first_moment[f,nmiss[r][f],:] * tmpc,0) \
                            / (self.rate_alpha[r]*(bb[r]+datum.scale))
                        tmpb = tmpb + frame.posterior[f] * np.sum(state.pos_first_moment[f,nmiss[r][f],:] * digamma(tmpc),0)
                        
                        for l from 0 <= l < f:
                            if not datum.missing[l,r]:
                                tmpa[0] = tmpa[0] + frame.posterior[f] * (self.rate_alpha[r,0]*bb[r,0]+datum.obs[l,r]) \
                                    / (self.rate_alpha[r,0]*(bb[r,0]+datum.scale/3.))
                                tmpb[0] = tmpb[0] + frame.posterior[f] * digamma(self.rate_alpha[r,0]*bb[r,0]+datum.obs[l,r])
                        for l from 3*datum.M+f <= l < datum.L:
                            if not datum.missing[l,r]:
                                tmpa[8] = tmpa[8] + frame.posterior[f] * (self.rate_alpha[r,8]*bb[r,8]+datum.obs[l,r]) \
                                    / (self.rate_alpha[r,8]*(bb[r,8]+datum.scale/3.))
                                tmpb[8] = tmpb[8] + frame.posterior[f] * digamma(self.rate_alpha[r,8]*bb[r,8]+datum.obs[l,r])

                    numbb[r] += tmpa - tmpb

            newbb = np.exp(numbb/den + digamma(self.rate_alpha*bb) - 1)

            relerr = np.mean(np.abs((newbb-bb)/bb))
            bb = newbb
            if relerr<reltol:
                break

        self.rate_beta = bb

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def update_alpha(self, data, states, frames):

        cdef bool optimized
        cdef long V
        cdef np.ndarray xo, x_init, x_final, notmissing

        # partition indices
        for datum,state in zip(data,states):
            for f in xrange(3):
                for r in xrange(4):
                    notmissing = np.array([1 if not np.any(datum.missing[3*m+f:3*m+3+f,r]) \
                        else 0 for m in xrange(datum.M)])
                    for s in xrange(self.S):
                        datum.indices[f][r][s] = np.where(np.logical_and(notmissing==1,state.pos_first_moment[f,:,s]>0))[0]

        # warm start for the optimization
        optimized = False
        xo = np.hstack(self.rate_alpha)
        V = xo.size
        x_init = xo.reshape(V,1)

        while not optimized:

            try:
                x_final, optimized = optimize_alpha(x_init, data, states, frames, self.rate_beta, self.start)
                if not optimized:
                    x_init = x_init*(1+0.1*(np.random.rand(V,1)-0.5))

            except ValueError:
                # if any parameter becomes Inf or Nan during optimization,
                # or if hessian is negative definite, re-optimize with a cold start
                x_init = x_init*(1+0.1*(np.random.rand(V,1)-0.5))

        self.rate_alpha = x_final.reshape(4,self.S)

    def __reduce__(self):
        return (rebuild_Emission, (self.logperiodicity, self.rate_alpha, self.rate_beta, self.start))

def rebuild_Emission(per, alpha, beta, start):
    e = Emission(start, 1)
    e.logperiodicity = per
    e.rate_alpha = alpha
    e.rate_beta = beta
    return e

def optimize_alpha(x_init, data, states, frames, beta, start):

    def F(x=None, z=None):

        if x is None:
            return 0, cvx.matrix(x_init)

        xx = np.array(x).ravel().astype(np.float64)

        if z is None:
            # compute likelihood function and gradient
            results = alpha_func_grad(xx, data, states, frames, beta, start)

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
            results = alpha_func_grad_hess(xx, data, states, frames, beta, start)

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
            eigs = np.linalg.eig(hess)
            if np.any(eigs[0]<0):
                raise ValueError
                # set hard constraint at parameters that give negative definite hessian
                #f = np.array([np.finfo(np.float32).max]).astype(np.float64)
                #eigval = np.abs(eigs[0])
                #hess = np.dot(eigs[1],np.dot(np.diag(eigval),eigs[1].T))
            hess = z[0] * hess

            return cvx.matrix(f), cvx.matrix(Df), cvx.matrix(hess)

    V = x_init.size
    # specify constraints on variables
    G = cvx.matrix(np.diag(-1*np.ones((V,), dtype='float')))
    h = cvx.matrix(np.zeros((V,1), dtype='float'))

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
cdef tuple alpha_func_grad(np.ndarray[np.float64_t, ndim=1] x, list data, \
list states, list frames, np.ndarray[np.float64_t, ndim=2] beta, str start):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, V, f, l, s
    cdef double func, psum
    cdef np.ndarray[np.int64_t, ndim=1] indices
    cdef np.ndarray[np.float64_t, ndim=1] Df, pos, bb
    cdef np.ndarray[np.float64_t, ndim=2] aa, dd, cc, ab, lbab, df

    V = x.size
    func = 0
    df = np.zeros((4,V/4),dtype=float)
    aa = np.empty((4,V/4), dtype=float)

    for r from 0 <= r < 4:
        aa[r] = x[r*V/4:(r+1)*V/4]
    ab = aa*beta
    dd = ab*np.log(beta) - gammaln(ab)
    lbab = np.log(beta) - digamma(ab)
    for datum,state,frame in zip(data,states,frames):

        cc = np.log(beta + datum.scale)
        for s from 0 <= s < V/4:
    
            for r from 0 <= r < 4:

                for f from 0 <= f < 3:

                    indices = datum.indices[f][r][s]
                    pos = state.pos_first_moment[f,indices,s]
                    psum = np.sum(pos)
                    bb = ab[r,s] + datum.total[f,indices,r]
                    func = func + frame.posterior[f] * ((dd[r,s]-ab[r,s]*cc[r,s])*psum + np.sum(pos*gammaln(bb)))
                    df[r,s] = df[r,s] + frame.posterior[f] * (np.sum(pos*digamma(bb)) + psum*(lbab[r,s]-cc[r,s]))

                    if s==0:

                        # add extra terms for first state
                        for l from 0 <= l < f:
                            if not datum.missing[l,r]:
                                func = func + frame.posterior[f] * (dd[r,0] \
                                    - ab[r,0]*np.log(beta[r,0]+datum.scale/3.) + gammaln(ab[r,0]+datum.obs[l,r]))
                                df[r,0] = df[r,0] + frame.posterior[f] * (lbab[r,0] \
                                    - np.log(beta[r,0]+datum.scale/3.) + digamma(ab[r,0]+datum.obs[l,r]))

                    if s==8:

                        # add extra terms for last state
                        for l from 3*datum.M+f <= l < datum.L:
                            if not datum.missing[l,r]:
                                func = func + frame.posterior[f] * (dd[r,8] \
                                    - ab[r,8]*np.log(beta[r,8]+datum.scale/3.) + gammaln(ab[r,8]+datum.obs[l,r]))
                                df[r,8] = df[r,8] + frame.posterior[f] * (lbab[r,8] \
                                    - np.log(beta[r,8]+datum.scale/3.) + digamma(ab[r,8]+datum.obs[l,r]))

    func = -1.*func
    Df = -1.*beta.ravel()*df.ravel()

    return func, Df

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef tuple alpha_func_grad_hess(np.ndarray[np.float64_t, ndim=1] x, list data, \
list states, list frames, np.ndarray[np.float64_t, ndim=2] beta, str start):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef long r, V, f, l, s
    cdef double func, psum, hftmp
    cdef np.ndarray[np.int64_t, ndim=1] indices
    cdef np.ndarray[np.float64_t, ndim=1] Df, pos, bb
    cdef np.ndarray[np.float64_t, ndim=2] aa, dd, cc, ab, lbab, df, Hf, diag, pab

    V = x.size
    func = 0
    df = np.zeros((4,V/4),dtype=float)
    Hf = np.zeros((V,V),dtype=float)
    diag = np.zeros((4,V/4), dtype=float)
    aa = np.empty((4,V/4), dtype=float)

    for r from 0 <= r < 4:
        aa[r] = x[r*V/4:(r+1)*V/4]
    ab = aa*beta
    dd = ab*np.log(beta) - gammaln(ab)
    lbab = np.log(beta) - digamma(ab)
    pab = polygamma(1, ab)
    for datum,state,frame in zip(data,states,frames):

        cc = np.log(beta + datum.scale)
        for s from 0 <= s < V/4:
    
            for r from 0 <= r < 4:

                for f from 0 <= f < 3:

                    indices = datum.indices[f][r][s]
                    pos = state.pos_first_moment[f,indices,s]
                    psum = np.sum(pos)
                    bb = ab[r,s] + datum.total[f,indices,r]
                    func = func + frame.posterior[f] * (np.sum(pos*gammaln(bb)) + (dd[r,s]-ab[r,s]*cc[r,s])*psum)
                    df[r,s] = df[r,s] + frame.posterior[f] * (np.sum(pos*digamma(bb)) + psum*(lbab[r,s]-cc[r,s]))
                    diag[r,s] = diag[r,s] + frame.posterior[f] * (np.sum(pos*polygamma(1, bb)) - psum*pab[r,s])

                    if s==0:

                        # add extra terms for first state
                        for l from 0 <= l < f:
                            if not datum.missing[l,r]:
                                func = func + frame.posterior[f] * (dd[r,0] \
                                    - ab[r,0]*np.log(beta[r,0]+datum.scale/3.) + gammaln(ab[r,0]+datum.obs[l,r]))
                                df[r,0] = df[r,0] + frame.posterior[f] * (lbab[r,0] \
                                    - np.log(beta[r,0]+datum.scale/3.) + digamma(ab[r,0]+datum.obs[l,r]))
                                diag[r,0] = diag[r,0] + frame.posterior[f] * (polygamma(1,ab[r,0]+datum.obs[l,r])-pab[r,0])

                    if s==8:

                        # add extra terms for last state
                        for l from 3*datum.M+f <= l < datum.L:
                            if not datum.missing[l,r]:
                                func = func + frame.posterior[f] * (dd[r,8] \
                                    - ab[r,8]*np.log(beta[r,8]+datum.scale/3.) + gammaln(ab[r,8]+datum.obs[l,r]))
                                df[r,8] = df[r,8] + frame.posterior[f] * (lbab[r,8] \
                                    - np.log(beta[r,8]+datum.scale/3.) + digamma(ab[r,8]+datum.obs[l,r]))
                                diag[r,8] = diag[r,8] + frame.posterior[f] * (polygamma(1,ab[r,8]+datum.obs[l,r])-pab[r,8])

    func = -1.*func
    Df = -1.*beta.ravel()*df.ravel()
    Hf[range(V),range(V)] = -1*beta.ravel()**2*diag.ravel()

    return func, Df, Hf

def learn_parameters(observations, codon_id, scales, missings, stop, restarts, mintol):

    cdef long restart, i
    cdef double scale, Lmax, L, dL, newL, reltol, starttime, totaltime
    cdef str start
    cdef list data, Ls, states, frames, ig
    cdef dict id
    cdef np.ndarray observation, missing
    cdef Data datum
    cdef Emission emission, best_emission
    cdef Transition transition, best_transition
    cdef State state
    cdef Frame frame

    data = [Data(observation, id, scale, missing) \
        for observation, id, scale, missing in zip(observations, codon_id, scales, missings)]
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
        emission.update_beta(data, states, frames, 1e-3)
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
            emission.update_beta(data, states, frames, 1e-3)
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

def infer_coding_sequence(observations, codon_id, scales, missings, transition, emission):

    cdef Data datum
    cdef State state
    cdef Frame frame
    cdef dict id
    cdef double scale
    cdef list data, states, frames
    cdef np.ndarray observation, missing

    data = [Data(observation, id, scale, missing) for observation,id,scale,missing in zip(observations,codon_id,scales,missings)]
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

