"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
history_len = 0

def save_anim(anim,label,dir,format):
    anim.save('{}/{}.{}'.format(dir,label,format))

def make_anim(y,t,Nt,typecolor):
    def animate(i):
        dt = t[2]-t[1]
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(thisx[1])
        history_y.appendleft(thisy[1])

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i*dt))
        return line, trace, time_text
    
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1  

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-(L+1), (L+1)), ylim=(-(L+1), (L+1)))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-',color='k', markerfacecolor = typecolor, lw=2)
    trace, = ax.plot([], [], '.-',color=typecolor, lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

    anim = animation.FuncAnimation(fig, animate,
                                    frames=Nt, interval=1000*(t[2]-t[1])*0.8, blit=True)
    return anim