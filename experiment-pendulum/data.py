# Dissipative Lagrangian Neural Networks|2023
# Takemori Masaki,Kamakura Yoshinari
# This software includes the work that is distributed in the Apache License 2.0 

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os
import concurrent

g = 9.8
L1 = 1
m1 = 1
rho = 0.5

def lagrangian_fn(coords):
    t1, w1 = np.split(coords,2)
    T = 0.5 * m1 * L1 ** 2 * w1 ** 2 
    V = - 1 * m1 * g * L1 * np.cos(t1)
    return T - V

def reighley_dissipation_fn(coords):
    q, qdot = np.split(coords,2)
    D = 0.5 * rho * qdot**2
    return D

def lagrangian_dynamics_fn(t, coords):
    qdot = coords[1]
    grad_L = autograd.grad(lagrangian_fn)(coords)
    grad_D = autograd.grad(reighley_dissipation_fn)(coords)
    hessian_L = autograd.hessian(lagrangian_fn)(coords)
    dL_dq = grad_L[0]
    dD_dqdot = grad_D[1]
    d2L_dq2 = hessian_L[0,0,0]
    d2L_dqdqdot = hessian_L[0,0,1]
    d2L_dqdot = hessian_L[0,1,1]
    q2dot = (-d2L_dqdqdot * qdot + dL_dq - dD_dqdot) / d2L_dqdot
    return [qdot, q2dot]


def get_lagrangian_trajectory(x=0,qf=1,qdf=20,t_span=[0,8], timescale=30, radius=None, y0=None, noise_std=0.1,**kwargs):
    # get initial state
    if y0 is None:
        y0 = np.concatenate([np.mod(((np.random.beta(1.3,1.3,1)-0.5) * 2*qf *np.pi),2*np.pi),(np.random.beta(0.8,0.8,1)-0.5)*2*qdf])

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    pend_ivp = solve_ivp(fun=lagrangian_dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, qdot = pend_ivp['y'][0], pend_ivp['y'][1]
    q = np.mod(q,2*np.pi)
    dydt = [lagrangian_dynamics_fn(None, y) for y in pend_ivp['y'].T]
    dydt = np.stack(dydt).T
    dq_dt, dqdot_dt = np.split(dydt,2)
    return q, qdot, dq_dt, dqdot_dt, t_eval

def get_DLNN_dataset(seed=0, samples=50, test_split=0.5,q=0.25,qd=0, **kwargs):

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []


    qf = np.tile(q,samples)
    qdf = np.tile(qd,samples)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        rets = executor.map(get_lagrangian_trajectory, range(samples),qf,qdf)

    for ret in rets :
        x, y, dx, dy, t = ret
        xs.append( np.stack([x,y]).T )
        dxs.append( np.stack( [dx, dy]).T )

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x','dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data

    return data

def get_lagrangian_trajectory_all(x=0,qf=1,qdf=20,t_span=[0,8], timescale=30, radius=None, y0=None, noise_std=0.1,**kwargs):
    # get initial state
    if y0 is None:
        y0 = np.concatenate([np.mod(((np.random.random(1)-0.5) * 2*qf *np.pi),2*np.pi),(np.random.random(1)-0.5)*2*qdf])

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    pend_ivp = solve_ivp(fun=lagrangian_dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, qdot = pend_ivp['y'][0], pend_ivp['y'][1]
    q = np.mod(q,2*np.pi)
    dydt = [lagrangian_dynamics_fn(None, y) for y in pend_ivp['y'].T]
    dydt = np.stack(dydt).T
    dq_dt, dqdot_dt = np.split(dydt,2)
    return q, qdot, dq_dt, dqdot_dt, t_eval

def get_DLNN_dataset_all(seed=0, samples=50, test_split=0.5,q=0.25,qd=0, **kwargs):

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []

    qf = np.tile(q,samples)
    qdf = np.tile(qd,samples)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        rets = executor.map(get_lagrangian_trajectory_all, range(samples),qf,qdf)
    
    for ret in rets :
        x, y, dx, dy, t = ret
        xs.append( np.stack([x,y]).T )
        dxs.append( np.stack( [dx, dy]).T )

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data

    return data