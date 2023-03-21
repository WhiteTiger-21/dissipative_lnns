# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os
import concurrent

g = 9.8
L1 = 1
L2 = 1
m1 = 1
m2 = 1
RHO = 0.5

##### ENERGY #####
def potential_energy(state):
    theta1,theta2,omega1,omega2 = np.split(state,4)
    V = - (m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return V


def kinetic_energy(state):
    theta1,theta2,omega1,omega2 = np.split(state,4)
    T1 = 0.5 * m1 * (L1 * omega1)**2
    T2 = 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 +
                    2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = T1 + T2
    return T

def lagrangian_fn(state):
    T = kinetic_energy(state)
    V = potential_energy(state)
    return T - V

def reighley_dissipation_fn(state):
    theta1,theta2,omega1,omega2 = np.split(state,4)
    D = 0.5 * RHO * (omega1**2 + omega2**2)
    return D


def lagrangian_dynamics_fn(t, coords):
    qdot = coords[2:]
    grad_L = autograd.grad(lagrangian_fn)(coords)
    hessian_L = autograd.hessian(lagrangian_fn)(coords)
    grad_D = autograd.grad(reighley_dissipation_fn)(coords)

    q2dot = (np.linalg.inv(hessian_L[0,2:,2:])) @ (grad_L[:2] - (hessian_L[0,:2,2:] @ qdot) - grad_D[2:])
    return np.hstack([qdot,q2dot])

def get_lagrangian_trajectory(x=0,qf=1,qdf=20,t_span=[0,8], timescale=30, radius=None, y0=None, noise_std=0.1,**kwargs):
    # get initial state
    if y0 is None:
        y0 = np.concatenate([np.mod(((np.random.beta(.85,8.5,2)-0.5) * 2*qf *np.pi),2*np.pi),(np.random.beta(0.8,0.8,2)-0.5)*2*qdf])

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    dp_ivp = solve_ivp(fun=lagrangian_dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, qdot = np.array([dp_ivp['y'][0], dp_ivp['y'][1]]),np.array([dp_ivp['y'][2], dp_ivp['y'][3]])
    q = np.mod(q,2*np.pi)
    dydt = [lagrangian_dynamics_fn(None, y) for y in dp_ivp['y'].T]
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
        xs.append( np.concatenate([x,y]).T )
        dxs.append( np.concatenate( [dx, dy]).T )

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data

    return data

def get_lagrangian_trajectory_all(x=0,qf=1,qdf=20,t_span=[0,8], timescale=30, radius=None, y0=None, noise_std=0.1,**kwargs):
    # get initial state
    if y0 is None:
        y0 = np.concatenate([np.mod(((np.random.random(2)-0.5) * 2*qf *np.pi),2*np.pi),(np.random.random(2)-0.5)*2*qdf])

    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    dp_ivp = solve_ivp(fun=lagrangian_dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, qdot = np.array([dp_ivp['y'][0], dp_ivp['y'][1]]),np.array([dp_ivp['y'][2], dp_ivp['y'][3]])
    q = np.mod(q,2*np.pi)
    dydt = [lagrangian_dynamics_fn(None, y) for y in dp_ivp['y'].T]
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
        xs.append( np.concatenate([x,y]).T )
        dxs.append( np.concatenate( [dx, dy]).T )

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data

    return data

def get_DLNN_dataset_random(seed=0, samples=50, test_split=0.2, **kwargs):

    data = {'meta': locals()}
    # randomly sample inputs
    np.random.seed(seed)

    xs, dxs = [], []
    for s in range(samples):
        if s % (samples//10) == 0 :
            print("{}/{}".format(s,samples))
        q1 = (np.random.rand()) * 2.0 * np.pi
        q2 = (np.random.rand()) * 2.0 * np.pi
        q1dot = (np.random.rand() - 0.5) * 2.0
        q2dot = (np.random.rand() - 0.5) * 2.0

        dx = lagrangian_dynamics_fn(None, np.array([q1, q2, q1dot, q2dot]))
        dq1_dt = dx[0]
        dq2_dt = dx[1]
        dq1dot_dt = dx[2]
        dq2dot_dt = dx[3]

        xs.append(np.array([q1, q2, q1dot, q2dot]))
        dxs.append(np.array([dq1_dt, dq2_dt, dq1dot_dt, dq2dot_dt]))

    data['x'] = np.stack(xs)
    data['dx'] = np.stack(dxs)

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data

    return data
