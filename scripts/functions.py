import numpy as np

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = (act*sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred))).sum()
    ll = ll * -1.0/len(act)
    return ll


def sample(dataframe, size):
    rows = np.random.choice(dataframe.index.values, size)
    sam_tr = dataframe.ix[rows]
    return sam_tr