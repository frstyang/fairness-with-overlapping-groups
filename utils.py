import numpy as np
import pandas as pd
import itertools

def binarize(X):
    # takes in a 2D numpy array and binarizes each column
    # by checking whether each entry is >= the mean of its column
    means = np.mean(X, axis=0)
    return (X >= means).astype(np.int32)

def normalize(X):
    # normalize each column of X by the mean and std
    return (X-X.mean(axis=0))/np.maximum(1e-5, X.std(axis=0))

def cat2onehot(vec):
    # vec is a vector where each entry is one of N categories
    # converts to a vec.shape[0] X N matrix, one hot encodes vec
    cat2ind = {cat: i for i,cat in enumerate(set(vec))}
    ind2cat = {i:cat for cat, i in cat2ind.items()}
    cats = []
    for i in range(len(cat2ind)):
        cats.append(ind2cat[i])
    onehot_mat = np.zeros((vec.shape[0], len(cat2ind)))
    for i in range(vec.shape[0]):
        onehot_mat[i, cat2ind[vec[i]]] = 1
    return onehot_mat, cats

def process_df_cats(df):
    types = [df[col].dtype for col in df.columns]
    num_cols = [col for i,col in enumerate(df.columns) if types[i] in [int, float]]
    obj_cols = [col for i,col in enumerate(df.columns) if types[i] == 'O']
    obj_df = df[obj_cols].copy()
    num_df = df[num_cols].copy()
    
    for col in obj_cols:
        ohm, cats = cat2onehot(obj_df[col].values)
        num_df[cats] = ohm
        
    return num_df
    
def intm(Xp):
    # takes in a binarized protected attribute matrix
    # returns base intersectional groups matrix
    groups = np.unique(Xp, axis=0)
    return (Xp[:, None, :] == groups).all(axis=2).astype(int)
    
    
def int_gerry(Xp):
    # takes in a binarized protected attribute matrix
    # returns all possible gerrymandering groups (any group
    # defined by a subset of the binary attributes)
    int_gerry_Xp = np.hstack((Xp, 1-Xp))
    m = Xp.shape[1]
    for k in range(2, m+1):
        for subset in itertools.combinations(range(m), k):
            sub_Xp = Xp[:, subset]
            intm_sub_Xp = intm(sub_Xp)
            int_gerry_Xp = np.hstack((int_gerry_Xp, intm_sub_Xp))
    return int_gerry_Xp

def calc_ind_viol(preds, Xp):
    # calculate independent fairness violation
    # of predictions preds on protected att mat Xp
    n = Xp.shape[0]
    gs = Xp.sum(axis=0)
    pmean = preds.mean()
    gmeans = (preds @ Xp)/gs 
    gviols = gmeans - pmean
    return max(np.max(np.abs(gviols)), np.max(np.abs(gviols*gs/(n-gs))))

def calc_gerry_viol(preds, gerry_Xp):
    #v^T(I-11^T/n)Xp
    n,m = gerry_Xp.shape
    pmean = preds.mean()
    return np.max(np.abs((preds@gerry_Xp/n-pmean*gerry_Xp.mean(axis=0))))

def calc_gerry_viol2(preds, Xp):
    # calculate gerrymandering fairness violation
    # iterates over all possible groups defined by a subset of attributes
    # the violation is scaled by the proportion of the group
    n,m = Xp.shape
    max_abs_viol = 0
    pmean = np.mean(preds)
    for k in range(1,m+1):
        for subset in itertools.combinations(range(m), k):
            subset_Xp = Xp[:, subset]
            unique_vecs = np.unique(subset_Xp, axis=0)
            for vec in unique_vecs:
                g_ind = (subset_Xp == vec).all(axis=1).astype(int)
                gmean = g_ind@preds/n
                gerry_viol = np.abs((pmean - gmean)*(g_ind.sum())/n)
                if gerry_viol > max_abs_viol:
                    max_abs_viol = gerry_viol
    return max_abs_viol

def calc_acc(preds, y):
    return (y*preds + (1-y)*(1-preds)).mean()

def tr_test_split(X, Xp, y, frac=0.7, random=True):
    tr_n = int(X.shape[0]*frac)
    X = X.copy()
    Xp = Xp.copy()
    y = y.copy().astype(int)
    if random:
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        Xp = Xp[perm]
        y = y[perm]
    tr_X, test_X = X[:tr_n], X[tr_n:]
    tr_Xp, test_Xp = Xp[:tr_n], Xp[tr_n:]
    tr_y, test_y = y[:tr_n], y[tr_n:]
    return (tr_X, tr_Xp, tr_y), (test_X, test_Xp, test_y)

def product_dicts(list_dict):
    keys = list(list_dict.keys())
    values = [list_dict[key] for key in keys]
    dicts = []
    for tup in itertools.product(*values):
        dicts.append(dict(zip(keys, tup)))
    return dicts

