import numpy as np
import pandas
import gerryfair
class GerryFair:
    def __init__(self, C, T, nu):
        self.C = C 
        self.T = T
        self.nu = nu 
        self.name='GerryFair'

    def train(self, X, Xp, y):
        fair_model = gerryfair.model.Model(C=self.C, printflag=False, gamma=self.nu, fairness_def='DP')
        fair_model.set_options(max_iters=self.T)
        X = pandas.DataFrame(X)
        Xp = pandas.DataFrame(Xp)
        y = pandas.Series(y)
        fair_model.train(X, Xp, y)
        self.fair_model = fair_model
        return 

    def predict(self, X, Xp):
        X = pandas.DataFrame(X)
        return np.array(self.fair_model.predict(X))