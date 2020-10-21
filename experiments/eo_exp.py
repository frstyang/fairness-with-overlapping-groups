import numpy as np
from algos import groupfair, regularizer
from utils import calc_acc, calc_ind_viol, calc_eo_viol

methods = [regularizer.Regularizer, groupfair.Plugin, groupfair.WERM]
params_list = [{'rho':np.logspace(-1,5, 30), 'T': [2000], 'lr':[0.001], 'nlayers': [1], 'fairness': ['EO']}, 
               {'B':[50], 'nu':np.logspace(-4,0,20), 'T':[10000], 'lr': [0.01], 'fairness': ['EO']},
               {'B':[50], 'nu':np.logspace(-4,0,20), 'T':[10000], 'lr': [0.01], 'fairness': ['EO']}]

metrics_list = [[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y))]
                ]
