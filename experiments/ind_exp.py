import numpy as np
from algos import groupfair, regularizer
from utils import calc_acc, calc_ind_viol

methods = [regularizer.Regularizer, groupfair.Plugin, groupfair.WERM]
params_list = [{'rho':np.logspace(-1,5, 30), 'T': [7500], 'lr':[0.01], 'nlayers': [1]}, 
               {'B':[50], 'nu':np.logspace(-3,0,20), 'T':[10000], 'lr': [0.1]},
               {'B':[50], 'nu':np.logspace(-3,0,20), 'T':[10000], 'lr': [0.1]}]

metrics_list = [[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_ind_viol(p, xp))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_ind_viol(p, xp))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_ind_viol(p, xp))]
                ]


