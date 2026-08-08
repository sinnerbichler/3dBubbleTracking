#!/usr/bin/env python

####### import library
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

####### customize plot
plt.style.use("seaborn-bright")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.monospace"] = "Times New Roman"
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["lines.linewidth"] = 1
rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
rainbow = [
    (0.278431, 0.278431, 0.858824),
    (0, 0, 0.360784),
    (0, 1, 1),
    (0, 0.501961, 0),
    (1, 1, 0),
    (1, 0.380392, 0),
    (0.419608, 0, 0),
    (0.878431, 0.301961, 0.301961),
]
cm = LinearSegmentedColormap.from_list("aname", rainbow, N=128)

colors = ["b", "lime", "r", "purple", "orange", "dodgerblue", "fuchsia", "yellow", "g"]


####### Read file
print("Begin reading of HDF5 file: full_traj_tracers.h5 ")

N = 8  # Number of trajectories to read

xfile = h5py.File("full_traj_tracers.h5", "r")
traj = xfile["traj3d"][0:N]  # read dataset from particle 0 to particle N
num_x = np.shape(traj)[0]
print(
    "Number of particles=%d, Time steps=%d, Number of components of the Dataset = %d"
    % (np.shape(traj)[0], np.shape(traj)[1], np.shape(traj)[2])
)
xfile.close()


####### File Structure:                                                                                                                   ####### -------------------------------------------
####### traj[ParticleNumber, TimeSteps, Components]
####### ParticleNumber: particle index that can go from 0 to 327679 (the total number of dumped trajectories)
####### TimeSteps: tmin=0 tmax=2000
####### Components: index that can go from 0 to 17 (see below)
####### -------------------------------------------
####### Components: 0,1,2 = x,y,z
####### Components: 3,4,5 = ux,uy,uz                                                                                                      ####### Components: 6,7,8 = ax,ay,az                                                                                                      ####### Components:  9,10,11 = dvx/dx,dvy/dx,dvz/dx                                                                                       ####### Components: 12,13,14 = dvx/dy,dvy/dy,dvz/dy                                                                                       ####### Components: 15,16,17 = dvx/dz,dvy/dz,dvz/dz

####### E.g. To read the ux component of particle 'i' from time step tmin to time step tmax:
####### traj[i,tmin:tmax,3]


""" EXAMPLES """

tmin = 0  # Choosing to look at the entire signal; starting from 0 to the total time steps, 2000.
tmax = 2000

####### Plot particles trajectories from timestep tmin to timestep tmax
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 12))

for ParticleNumber in range(N):
    xp = traj[ParticleNumber, tmin:tmax, 0]
    yp = traj[ParticleNumber, tmin:tmax, 1]
    zp = traj[ParticleNumber, tmin:tmax, 2]
    if N <= len(colors):
        ax.scatter(
            xp,
            yp,
            zp,
            label="Particle number %d" % (ParticleNumber),
            color=colors[ParticleNumber],
        )
    else:
        ax.scatter(xp, yp, zp, label="Particle number %d" % (ParticleNumber))
    ax.legend()
    # ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

ax.view_init(60, 75)
plt.savefig("trajectories.png")
plt.show()
plt.close()


######## Velocity trajectories
for ParticleNumber in range(N):
    ux = traj[ParticleNumber, tmin:tmax, 3]
    # uy = traj[ParticleNumber,tmin:tmax,4]
    # uz = traj[ParticleNumber,tmin:tmax,5]
    if N <= len(colors):
        plt.plot(
            ux,
            label="Particle number %d" % (ParticleNumber),
            color=colors[ParticleNumber],
        )
    else:
        plt.plot(ux, label="Particle number %d" % (ParticleNumber))
    plt.legend()

plt.xlabel("TimeSteps")
plt.ylabel("ux")
plt.savefig("velocity.png")
plt.close()


######## Acceleration
for ParticleNumber in range(N):
    ax = traj[ParticleNumber, tmin:tmax, 6]
    # ay = traj[ParticleNumber,tmin:tmax,7]
    # az = traj[ParticleNumber,tmin:tmax,8]
    if N <= len(colors):
        plt.plot(
            ax,
            label="Particle number %d" % (ParticleNumber),
            color=colors[ParticleNumber],
        )
    else:
        plt.plot(ax, label="Particle number %d" % (ParticleNumber))
    plt.legend()

plt.xlabel("TimeSteps")
plt.ylabel("ax")
plt.savefig("acceleration.png")
plt.close()


######## Gradients component of one particle
ParticleNumber = 0

labelgrads = [
    "dux/dx",
    "duy/dx",
    "duz/dx",
    "dux/dy",
    "duy/dy",
    "duz/dy",
    "dux/dz",
    "duy/dz",
    "duz/dz",
]
for grad in range(9):
    plt.plot(
        traj[ParticleNumber, tmin:tmax, 9 + grad],
        label=labelgrads[grad],
        color=colors[grad],
    )

plt.legend()
plt.xlabel("TimeSteps")
plt.ylabel("Gradients along Particle Number=%d" % (ParticleNumber))
plt.savefig("gradients.png")
plt.close()
