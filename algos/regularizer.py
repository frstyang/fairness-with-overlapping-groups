import numpy as np
import torch
import torch.nn as nn

def fair_reg(preds, Xp):
    Xp = torch.cat((Xp, 1-Xp), dim=1)
    viol = preds.mean()-(preds@Xp)/torch.max(Xp.sum(axis=0), torch.ones(Xp.shape[1])*1e-5)
    return (viol**2).mean()

class Regularizer:
    def __init__(self, rho, T, lr=0.01, nlayers=1, fairness='DP'):
        self.rho=rho
        self.T=T
        self.lr=lr
        self.fairness=fairness
        self.nlayers = nlayers
        self.name = 'Regularizer'
        
    def construct_learner(self, d):
        layers = []
        for i in range(self.nlayers-1):
            layers.append(nn.Linear(d,128))
            layers.append(nn.ReLU())
            d=128
        layers.append(nn.Linear(d,1))
        layers.append(nn.Sigmoid())
        self.learner = nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.learner.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        self.optim, milestones=[self.T//2,self.T//2+self.T//4], gamma=0.5)
        
    def train(self, X, Xp, y, evaluate=False):
        bce_loss = nn.BCELoss(reduction='mean')
        if not hasattr(self, 'learner'):
            self.construct_learner(X.shape[1])
        
        X = torch.Tensor(X)
        Xp = torch.Tensor(Xp)
        y = torch.Tensor(y)
        for t in range(self.T):
            preds = self.learner(X).flatten()
            bce = bce_loss(preds, y)
            if self.fairness == 'EO':
                reg = fair_reg(preds[y==1], Xp[y==1])
            if self.fairness == 'DP':
                reg = fair_reg(preds, Xp)
            loss = bce + self.rho*reg
            if evaluate:
                print(f't:{t}, bce:{bce.item():.7f}, reg:{reg.item():.7f}, loss:{loss.item():.7f}')
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            #self.scheduler.step()
        return 
    
    def predict(self, X, Xp):
        X = torch.Tensor(X)
        return self.learner(X).flatten().detach().numpy()