
from Queue import PriorityQueue
from scipy import stats
import numpy as np


def simulateInhomPoisson(branching, kernel):
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
    t = kernel.rvs(n)
    return np.sort(t)


def simulateHomPoisson(T, mu, discrete=False):
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
    if discrete:
        return np.floor(T * u).astype(int)
    return T * u


def simulateHawkes(hawkesProcess, T, discrete=False):
    '''
    simulate univariate unmarked or marked Hawkes process.

    For marked process, mu has to be a vector and branching, tau
    are matrices.

    Args:
      hawkesProcess: process to simulate
      T: observation period
      discrete: model integer time steps or continuous time

    Returns:
      t (ndarray): array containing event times
      mark (ndarray): marks of events, only for marked process
      parent (ndarray(int)): array of parents for each event, -1 for background
      cluster (ndarray(int)): cluster to which each event belongs
    '''
    n = hawkesProcess.n_marks
    if discrete:
        T = int(T)
    q = PriorityQueue()
    # immigrants
    for mark, rate in enumerate(hawkesProcess.mu):
        for t in simulateHomPoisson(T, rate, discrete):
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
            for dt in simulateInhomPoisson(hawkesProcess.b[mark, childMark], hawkesProcess.kernel.dist(mark, childMark)):
                t = t0 + dt
                if t < T:
                    q.put((t, childMark, i, cluster))
        i += 1
    if not result:
        return [np.array([]) for i in range(4)]
    return tuple(np.array(r) for r in zip(*result))
