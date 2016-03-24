import numpy as np


def get_clusters_from_parents(parent):
    '''
    Get cluster labels from parent relationships.

    Args:
      parent: sequence containing index of parent event
              for each event. -1 denotes background events.

    Returns:
      label (ndarray): list of cluster labels.
    '''
    def visit(i, c):
        if not visited[i]:
            visited[i] = True
            label[i] = c
            for n in neighbors[i]:
                visit(n, c)
            return True
        return False

    n = len(parent)
    neighbors = [[] for i in range(n)]
    for i, p in enumerate(parent):
        if p > -1:
            neighbors[i].append(p)
            neighbors[p].append(i)
    label = -np.ones(n, dtype=int)
    visited = [False] * n
    c = 0
    for i in range(n):
        if visit(i, c):
            c += 1
    return label


def cluster_events(z):
    '''
    get most probable event trees from Hawkes latent variables.

    Args:
      z: latent connections as returned by HawkesProcess.transform

    Returns:
      parent, cluster:
        parent: array with indices of most likely parent,
                -1 for background
        cluster: array with cluster assignments. Clusters are numbered
                 starting from zero.
    '''
    parent = z.argmax(axis=0)
    parent[parent == z.shape[1]] = -1
    cluster = get_clusters_from_parents(parent)
    return parent, cluster
