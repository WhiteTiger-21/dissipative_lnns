# Dissipative Lagrangian Neural Networks|2023
# Takemori Masaki,Kamakura Yoshinari
# This software includes the work that is distributed in the Apache License 2.0 
import torch
import numpy as np

import os, sys,copy
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP

class DLNN(torch.nn.Module):

    def __init__(self, model):
        super(DLNN, self).__init__()
        self.differentiable_model = model # nn-model

    def forward(self, x):
        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def time_derivative(self, x, t=None):
        x = torch.autograd.Variable(x, requires_grad=True)
        q, qdot = x.T
        xx = torch.stack((q, qdot), dim=-2).T
        L, D = self.forward(xx)

        dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True)[0]
        dL_dqdot = torch.autograd.grad(L.sum(), qdot, create_graph=True)[0]

        dD_dqdot = torch.autograd.grad(D.sum(), qdot, create_graph=True)[0]

        d2L_dqdqdot = torch.autograd.grad(dL_dq.sum(), qdot, create_graph=True)[0]
        d2L_dqdot2 = torch.autograd.grad(dL_dqdot.sum(), qdot, create_graph=True)[0]

        q2dot2 = (-d2L_dqdqdot * qdot + dL_dq - dD_dqdot) / d2L_dqdot2

        dx_dt = torch.stack((qdot, q2dot2), dim=-2).T

        return dx_dt