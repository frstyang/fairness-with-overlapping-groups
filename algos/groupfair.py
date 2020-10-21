import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class GroupFair:
    def __init__(self, B, nu, T, lr=0.1, gfair='ind', fairness='DP', lambda_update = 'subgradient'):
        # T = number of iterations to train for
        self.B = B
        self.nu = nu
        self.T = T
        self.gfair = gfair
        assert gfair in ['ind', 'gerry']
        self.fairness = fairness
        assert fairness in ['DP', 'EO']
        self.lr=lr
        assert lambda_update in ['subgradient', 'expgradient']
        self.lambda_update = lambda_update
        self.predictors = []
        self.lambdas = []
        self.viols = []
        return
    
    def min_oracle(self, lambda_, X, Xp, y):
        return
    
    def update_lambda(self, lambda_, viols, t):
        lambda_next = lambda_.copy()
        if self.lambda_update == 'subgradient':
            lambda_next[0] += self.lr*(viols-self.nu)/np.sqrt(t+1)
            lambda_next[1] += self.lr*(-viols-self.nu)/np.sqrt(t+1)
            lambda_next = np.maximum(np.minimum(lambda_next, self.B), 0)
        if self.lambda_update == 'expgradient':
            theta_next = self.thetas[-1].copy()
            theta_next[0] += self.lr*(viols-self.nu)/np.sqrt(t+1)
            theta_next[1] += self.lr*(-viols-self.nu)/np.sqrt(t+1)
            self.thetas.append(theta_next)
            lambda_next = self.theta_to_lambda(theta_next)
        return lambda_next
    
    def calc_viols(self, h, X, Xp):
        preds = h(X, Xp)
        preds_mean = preds.mean()
        if self.gfair == 'ind':
            preds_gmeans = (preds@Xp)/np.maximum(Xp.sum(axis=0), 1e-5)
            return preds_mean - preds_gmeans
        if self.gfair == 'gerry':
            n=X.shape[0]
            return Xp.mean(axis=0)*preds_mean - preds@Xp/n
    
    def theta_to_lambda(self, theta):
        exp_theta = np.exp(theta)
        return self.B*exp_theta/(exp_theta.sum()+1)
    
    def train(self, X, Xp, y, evaluate=False):
        n,m = Xp.shape
        if len(self.predictors) == 0:
            if self.lambda_update == 'subgradient':
                self.lambdas.append(np.zeros((2,m)))
            if self.lambda_update == 'expgradient':
                self.thetas = [np.zeros((2,m))]
                self.lambdas.append(self.theta_to_lambda(self.thetas[-1]))
        for t in range(self.T):
            lambda_t = self.lambdas[-1]
            h_t = self.min_oracle(lambda_t, X, Xp, y)
            preds = h_t(X, Xp)
            if self.fairness == 'DP':
                viols = self.calc_viols(h_t, X, Xp)
            if self.fairness == 'EO':
                viols = self.calc_viols(h_t, X[y==1], Xp[y==1])
            self.viols.append(viols)
            lambda_t_plus_1 = self.update_lambda(lambda_t, viols, t)
            
            self.predictors.append(h_t)
            self.lambdas.append(lambda_t_plus_1)
            if evaluate:
                print(t, 'train violations', viols, 'train acc', (preds*y+(1-preds)*(1-y)).mean())
            
        return self.predictors, self.lambdas
    
    def predict(self, X, Xp):
        predictions = np.mean([predictor(X, Xp) for predictor in self.predictors], axis=0)
        return predictions

class Plugin(GroupFair):
    def __init__(self, **kwargs):
        super(Plugin, self).__init__(**kwargs)
        self.name='Plugin'

    @ignore_warnings(category=ConvergenceWarning)
    def min_oracle(self, lambda_, X, Xp, y):
        if self.gfair == 'ind':
            pi_hat = Xp.mean(axis=0)
        if self.fairness == 'EO':
            pi_y_inv = 1/max(1e-5,np.mean(y))
            gp_pi_y = (y@Xp)/np.maximum(Xp.sum(axis=0), 1e-5)
            gp_pi_y_inv = 1/np.maximum(gp_pi_y, 1e-5)

        if not hasattr(self, 'eta'):
            lr = LogisticRegression(max_iter=1000, fit_intercept=True).fit(X, y)
            self.eta = lr.predict_proba
        def min_pred(input_X, input_Xp):
            n, m = input_Xp.shape
            cost = np.array([0,1,1,0])
            net_lambda = lambda_[0] - lambda_[1]
            if self.fairness == 'DP':
                if self.gfair == 'ind':
                    viol_weights = net_lambda.sum()-input_Xp@(net_lambda/pi_hat)
                if self.gfair == 'gerry':
                    viol_weights = Xp.mean(axis=0)@net_lambda - input_Xp@net_lambda
                cost = cost + viol_weights[:, None]*np.array([0,1,0,1])
            if self.fairness == 'EO':
                viol_weights = net_lambda.sum()*pi_y_inv - input_Xp@(net_lambda*gp_pi_y_inv/pi_hat)
                cost = cost + viol_weights[:, None]*np.array([0,0,0,1])
            cost = cost.reshape(n, 2,2)
            eta = self.eta(input_X)
            final_costs = (eta.reshape(n, 2, 1)*cost).sum(axis=1)
            return np.argmin(final_costs, axis=1)
        return min_pred

class WERM(GroupFair):
    def __init__(self, lr_type='logistic', **kwargs):
        super(WERM, self).__init__(**kwargs)
        self.name = 'WERM'
        self.lr_type = lr_type

    @ignore_warnings(category=ConvergenceWarning)
    def min_oracle(self, lambda_, X, Xp, y):
        n = X.shape[0]
        pi_hat = Xp.mean(axis=0)
        net_lambda = lambda_[0] - lambda_[1]
        if self.gfair == 'ind':
            if self.fairness == 'DP':
                viol_weights = net_lambda.sum()-Xp@(net_lambda/pi_hat)
                weights = viol_weights[:,None]*np.array([0,1]) + np.array([[0, 1],[1,0]])[y]
            if self.fairness == 'EO':
                pi_y_inv = 1/max(1e-5,np.mean(y))
                gp_pi_y = (y@Xp)/np.maximum(Xp.sum(axis=0), 1e-5)
                gp_pi_y_inv = 1/np.maximum(gp_pi_y, 1e-5)
                viol_weights = y*(net_lambda.sum()*pi_y_inv - Xp@(net_lambda*gp_pi_y_inv/pi_hat))
                weights = viol_weights[:, None]*np.array([0,1]) + np.array([[0, 1],[1,0]])[y]

        if self.gfair == 'gerry':
            viol_weights = pi_hat@net_lambda - Xp@net_lambda
        if self.lr_type=='logistic':
            weights = weights - weights.min(axis=1)[:, None]
            y_tilde = np.argmin(weights, axis=1)
            lr = LogisticRegression(max_iter=100, fit_intercept=True)\
                 .fit(X, y_tilde, sample_weight = weights[np.arange(n), 1-y_tilde])
            return lambda X, Xp: np.argmax(lr.predict_proba(X), axis=1)
        if self.lr_type == 'linear':
            target = weights[:, 0] - weights[:, 1]
            lr = LinearRegression(fit_intercept=True).fit(X, target)
            return lambda X, Xp: (lr.predict(X)>0).astype(int)