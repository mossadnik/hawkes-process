
import pandas as pd
import numpy as np


def treeLayout(children, d=1):
    '''Compute simple tree layout.'''

    sy = np.zeros(len(children))
    y = np.zeros(len(children))
    
    def getSize(root=0):        
        result = max(d, sum(getSize(c) for c in children[root]))
        sy[root] = result
        return result
    
    def setChildrenPosition(root=0):
        sizes = sy[children[root]]
        ymin = y[root] - .5 * sum(sizes)
        dy = 0
        for c, s in zip(children[root], sizes):
            y[c] = ymin + dy + .5 * s
            dy += s
            setChildrenPosition(c)

    getSize()
    setChildrenPosition()
    return y

def plotHawkes(t, parent, cluster, ax=None):
    '''Visualize an unmarked Hawkes process.'''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 5))
    events = pd.DataFrame(columns=['cluster', 'event', 'parent', 't'],
                 data={'t': t, 'parent': parent, 'cluster': cluster, 'event': np.arange(t.size, dtype=np.int)}
                ).sort_values(['cluster', 'parent', 't'])    
    for cluster, df in events.groupby('cluster')['event', 'parent', 't']:
        event = df.event.values
        # translate parent event to index
        evDict = {e: i for i, e in enumerate(event)}
        evDict.update({-1: -1})
        parent = df.parent.map(evDict).values
        children = [np.where(parent == i)[0] for i in range(event.size)]
        y = treeLayout(children)
        x = df.t.values
        for c, p in enumerate(parent):
            if p > -1:
                ax.plot([x[c], x[p]], [y[c], y[p]], 'k-', alpha=.4)
        ax.plot(x[1:], y[1:], '.', alpha=.7)    
        ax.plot(x[0], y[0], 'k.')