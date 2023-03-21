# Dissipative Lagrangian Neural Networks|2023
# Takemori Masaki,Kamakura Yoshinari
# This software includes the work that is distributed in the Apache License 2.0 
import torch
import numpy as np
from nn_models import MLP

import functorch

class DLNN(torch.nn.Module):
    def __init__(self, model):
        super(DLNN, self).__init__()  # Inherit the methods of the Module constructor
        self.mlp = model # Instantiate an MLP for learning the conservative component
        self.n = 0

    def forward(self, x): 
        y = self.mlp(x)
        return y

    def calculate(self,x):
        n = self.n
        qdot = x[n:]

        A = functorch.hessian(self.forward)(x)[0]
        BC = functorch.jacrev(self.forward)(x)

        qddot = torch.linalg.pinv(A[n:, n:]) @ (BC[0,:n] - A[:n, n:] @ qdot - BC[1,n:])
        xt = torch.cat([qdot,qddot])
        return xt

    def time_derivative(self, x):
        self.n = x.shape[-1]//2
        xv = torch.autograd.Variable(x, requires_grad=True)
        xt = functorch.vmap(self.calculate)(xv)
        return xt

    def t_forward(self, t, x):
        return self.forward(x)
