
from Queue import PriorityQueue
from scipy import stats
import numpy as np


def simulateInhomPoisson(branching, tau):
    '''
    simulate inhomogenous Poisson process with exponentially
    decaying rate.
    
    Args:
      branching (float): Average number of events, 0 <= branching < 1
      tau (float): Decay time of exponential time-dependency of rate
      
    Returns:
      t (ndarray(float)): event times
    '''
    n = stats.poisson(mu=branching).rvs()
    t = stats.expon(scale=tau).rvs(n)
    return np.sort(t)

def simulateHomPoisson(T, mu):
    '''
    simulate homogenous Poisson process.
    
    Args:
      T (float): observation time
      mu (float): event rate
      
    Returns:
      t (ndarray(float)): event times
    '''
    n = stats.poisson(mu*T).rvs()
    u = np.sort(np.random.rand(n))
    return T * u

def simulateHawkes(T, mu, branching, tau):
    '''
    simulate univariate unmarked or marked Hawkes process.
    
    For marked process, mu has to be a vector and branching, tau
    are matrices.
    
    Args:
      T (float): observation time
      mu (float): background event creation rate
      branching (float): branching ratio of the process, 0 <= branching < 1
      tau (float): exponential decay time of excited processes
      
    Returns:
      t (ndarray): array containing event times
      mark (ndarray): marks of events, only for marked process
      parent (ndarray(int)): array of parents for each event, -1 for background
      cluster (ndarray(int)): cluster to which each event belongs
    '''    
    mu = np.atleast_1d(mu)
    n = mu.size
    branching = np.atleast_2d(branching)
    if branching.size == 1:
        branching = branching[0][0] * np.ones((n, n))
    tau = np.atleast_2d(tau)
    if tau.size == 1:
        tau = tau[0][0] * np.ones((n, n))
    
    q = PriorityQueue()
    # immigrants
    for mark, rate in enumerate(mu):
        for t in simulateHomPoisson(T, rate):
            q.put((t, mark, -1, -1))
    i = 0
    iCluster = 0
    result = []
    # self-excited events
    while not q.empty():
        t0, mark, parent, cluster = q.get()
        if cluster < 0:
            cluster = iCluster
            iCluster += 1
        result.append((t0, mark, parent, cluster))
        for childMark in range(n):
            for dt in simulateInhomPoisson(branching[mark, childMark], tau[mark, childMark]):
                t = t0 + dt
                if t < T:
                    q.put((t, childMark, i, cluster))
        i += 1
    ret = tuple(np.array(r) for r in zip(*result))
    if n > 1:
        return ret
    else:
        # do not return mark for unmarked case
        return ret[0], ret[2], ret[3]
