import numpy as np
from algos import groupfair, gerfair
from utils import calc_acc, calc_gerry_viol, calc_gerry_viol2

methods = [groupfair.Plugin, groupfair.WERM, gerfair.GerryFair]
params_list = [{'B':[10], 'nu':np.logspace(-4,0,20), 'T':[10000], 'lr': [1], 'fairness': ['gerry'], 'lambda_update': ['subgradient']},
               {'B':[10], 'nu':np.logspace(-4,0,20), 'T':[10000], 'lr': [1], 'fairness': ['gerry'], 'lambda_update': ['subgradient']},
{'C':[15], 'nu':np.logspace(-3,0,20), 'T':[1500]}]

metrics_list = [
[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('gerry_viol', lambda p,x,xp,y: calc_gerry_viol(p, xp))],
[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('gerry_viol', lambda p,x,xp,y: calc_gerry_viol(p, xp))],
[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('gerry_viol', lambda p,x,xp,y: calc_gerry_viol2(p, xp))]
                ]


