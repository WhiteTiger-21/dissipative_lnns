# Dissipative Lagrangian Neural Networks|2023
# Takemori Masaki,Kamakura Yoshinari
# This software includes the work that is distributed in the Apache License 2.0 

import torch
import numpy as np

import os,sys
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

        q1, q2, q1dot, q2dot = x.T

        q1dash = (q1 + np.pi) % (2.0 * np.pi) - np.pi
        q2dash = (q2 + np.pi) % (2.0 * np.pi) - np.pi

        xx = torch.stack((q1dash, q2dash, q1dot, q2dot), dim=-2).T
        L, D = self.forward(xx)

        dL_dq1 = torch.autograd.grad(L.sum(), q1, create_graph=True)[0]
        dL_dq2 = torch.autograd.grad(L.sum(), q2, create_graph=True)[0]

        dL_dq1dot = torch.autograd.grad(L.sum(), q1dot, create_graph=True)[0]
        dL_dq2dot = torch.autograd.grad(L.sum(), q2dot, create_graph=True)[0]

        dD_dq1dot = torch.autograd.grad(D.sum(), q1dot, create_graph=True)[0]
        dD_dq2dot = torch.autograd.grad(D.sum(), q2dot, create_graph=True)[0]

        d2L_dq1_dq1dot = torch.autograd.grad(dL_dq1.sum(), q1dot, create_graph=True)[0]
        d2L_dq1_dq2dot = torch.autograd.grad(dL_dq1.sum(), q2dot, create_graph=True)[0]

        d2L_dq2_dq1dot = torch.autograd.grad(dL_dq2.sum(), q1dot, create_graph=True)[0]
        d2L_dq2_dq2dot = torch.autograd.grad(dL_dq2.sum(), q2dot, create_graph=True)[0]

        d2L_dq1dot_dq1dot = torch.autograd.grad(dL_dq1dot.sum(), q1dot, create_graph=True)[0]
        d2L_dq1dot_dq2dot = torch.autograd.grad(dL_dq1dot.sum(), q2dot, create_graph=True)[0]
        d2L_dq2dot_dq2dot = torch.autograd.grad(dL_dq2dot.sum(), q2dot, create_graph=True)[0]

        A11 = d2L_dq1dot_dq1dot
        A12 = d2L_dq1dot_dq2dot
        A21 = A12
        A22 = d2L_dq2dot_dq2dot

        DET_A = A11 * A22 - A12 * A21
        AINV11 = A22 / DET_A
        AINV12 = -A12 / DET_A
        AINV21 = -A21 / DET_A
        AINV22 = A11 / DET_A

        B11 = d2L_dq1_dq1dot
        B12 = d2L_dq2_dq1dot
        B21 = d2L_dq1_dq2dot
        B22 = d2L_dq2_dq2dot

        AINVB11 = AINV11 * B11 + AINV12 * B21
        AINVB12 = AINV11 * B12 + AINV12 * B22
        AINVB21 = AINV21 * B11 + AINV22 * B21
        AINVB22 = AINV21 * B12 + AINV22 * B22

        q1dotdot = -(AINVB11 * q1dot + AINVB12 * q2dot) + AINV11 * (dL_dq1 - dD_dq1dot) + AINV12 * (dL_dq2 - dD_dq2dot)
        q2dotdot = -(AINVB21 * q1dot + AINVB22 * q2dot) + AINV21 * (dL_dq1 - dD_dq1dot) + AINV22 * (dL_dq2 - dD_dq2dot)

        dx_dt = torch.stack((q1dot, q2dot, q1dotdot, q2dotdot), dim=-2).T

        return dx_dt
