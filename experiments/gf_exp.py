import numpy as np
from algos import groupfair, gerfair
from utils import calc_acc, calc_gerry_viol, calc_gerry_viol2

methods = [gerfair.GerryFair]
params_list = [{'C':[15], 'nu':np.logspace(-3,0,20), 'T':[500]}]

metrics_list = [
[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('gerry_viol', lambda p,x,xp,y: calc_gerry_viol2(p, xp))]
                ]


