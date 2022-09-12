"""
Dynamic structural analysis for free vibration of a pinned beam 
with initial velocity
"""
from logging import WARNING
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsX import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode',dest='mode',default=1,
                    help='Vibration mode number.')

args = parser.parse_args()
r = int(args.mode)

comm = MPI.COMM_WORLD

beam = [#### quad mesh ####
        "plate_2_10_quad_1_5.xdmf",
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",
        "plate_2_10_quad_20_100.xdmf",
        "plate_2_10_quad_40_200.xdmf",
        "plate_2_10_quad_80_400.xdmf",]
        
filename = "../../mesh/mesh-examples/clamped-RM-plate/"+beam[2]
with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

E = 1e7
nu = 0.0
h = 0.3
width = 2.
length = 10.

rho = 10.

PI = np.pi

C = 0.1
Ix = width*h**3/12
lam = r*PI/length
omega = (r*PI)**2*np.sqrt(E*Ix/(rho*width*h*length**4))

class u_exact:

    def __init__(self):
        self.t = 0.

    def eval(self, x):
        return (np.full(x.shape[1], 0.0), 
                np.full(x.shape[1], 0.0), 
                -C*np.sin(omega*self.t)*np.sin(lam*x[0]))

# udot = d(u)/d(t)
class udot_exact:

    def __init__(self):
        self.t = 0.

    def eval(self, x):
        return (np.full(x.shape[1], 0.0), 
                np.full(x.shape[1], 0.0), 
                -C*omega*np.cos(omega*self.t)*np.sin(lam*x[0]))

# thetadot_y = - d(udot_z)/d(x)
class thetadot_exact:
    def __init__(self):
        self.t = 0.

    def eval(self, x):
        return (np.full(x.shape[1], 0.0), 
                -C*lam*omega*np.sin(omega*self.t)*np.cos(lam*x[0]),
                np.full(x.shape[1], 0.0))

# uddot = d(udot)/d(t)
class uddot_exact:

    def __init__(self):
        self.t = 0.

    def eval(self, x):
        return (np.full(x.shape[1], 0.0), 
                np.full(x.shape[1], 0.0), 
                C*omega**2*np.sin(omega*self.t)*np.sin(lam*x[0]))

# Time-stepping parameters
T       = 2.0/r
Nsteps  = 200
dt = T/Nsteps
# starting time
t = 0.0

element_type = "CG2CG1"
element = ShellElement(
                mesh,
                element_type,
                )
W = element.W

w = Function(W)
u, theta = split(w)
dw = TestFunction(W)
du_mid,dtheta = split(dw)

# Quantities from the previous time step
w_old = Function(W)
u_old, theta_old = split(w_old)
wdot_old = Function(W)
udot_old, thetadot_old = split(wdot_old)

# Set up the time integration scheme
u_mid = Constant(mesh, 0.5)*(u_old+u)
theta_mid = Constant(mesh, 0.5)*(theta_old+theta)
w_mid = Constant(mesh, 0.5)*(w_old+w)
udot = Constant(mesh, 2/dt)*u - Constant(mesh, 2/dt)*u_old - udot_old
uddot = (udot - udot_old)/dt
thetadot = Constant(mesh, 2/dt)*theta - \
            Constant(mesh, 2/dt)*theta_old - thetadot_old
thetaddot = (thetadot - thetadot_old)/dt

w_temp = Function(W)
# Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = DynamicElasticModel(mesh, w_temp, material_model.CLT)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)

# Elastic energy contribution to the residual:
f_0 = Constant(mesh, (0.0,0.0,0.0)) # unforced vibration

# if we change ALPHA to 1/2 instead of replacing w_temp with w_mid,
# the results would be the same
ALPHA = 1
dWint = elastic_model.weakFormResidual(ALPHA, elastic_energy, w_temp, dw, f_0)
dWint_mid = ufl.replace(dWint, {w_temp: w_mid})

# Inertial contribution to the residual:
dWmass = elastic_model.inertialResidual(rho,h,uddot,thetaddot)
F = dWmass + dWint_mid


######### Pinned boundary condition on both ends of the plate ##########
# Define BCs geometrically
# The plate is pinned on both ends with the x-component of displacement unconstrained,
# to minimize the effect of membrane stress; the y-component of displacement is fixed
# to zero for the other two edges of the plate to mimic the Euler-Bernoulli beam deformation -
# - though there should not be obvious effect of this constraint.
locate_BC1 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                        lambda x: np.isclose(x[0], 0., atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                        lambda x: np.isclose(x[0], length, atol=1e-6))
locate_BC3 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                        lambda x: np.isclose(x[1], 0., atol=1e-6))
locate_BC4 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                        lambda x: np.isclose(x[1], width, atol=1e-6))
locate_BC5 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()[0]), 
                        lambda x: np.isclose(x[0], 0., atol=1e-6))
locate_BC6 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()[0]), 
                        lambda x: np.isclose(x[0], length, atol=1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)
     
bcs = [dirichletbc(ubc, locate_BC1, W.sub(0).sub(1)),
        dirichletbc(ubc, locate_BC2, W.sub(0).sub(1)),
        dirichletbc(ubc, locate_BC3, W.sub(0).sub(1)),
        dirichletbc(ubc, locate_BC4, W.sub(0).sub(1)),
        dirichletbc(ubc, locate_BC5, W.sub(0).sub(2)),
        dirichletbc(ubc, locate_BC6, W.sub(0).sub(2)),
       ]


# Set up the time-stepping
time = np.linspace(0, T, Nsteps+1)
L2_error = np.zeros((Nsteps+1,))
# the location of interest for outputs
x_ = length/(2**r)
coord = [x_, 1., 0.]
cell_id = getCellID(coord, mesh)
u_output = np.zeros((Nsteps+1,))
u_output_exact = -C*np.sin(lam*x_)*np.sin(omega*time)

PATH = "solutions/pinned_plate/"
xdmf_file = XDMFFile(comm, PATH+"displacement.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file_exact = XDMFFile(comm, PATH+"exact_displacement.xdmf", "w")
xdmf_file_exact.write_mesh(mesh)
xdmf_file_rotation = XDMFFile(comm, PATH+"rotation.xdmf", "w")
xdmf_file_rotation.write_mesh(mesh)

# Set up the initial condition
udot0 = udot_exact()
udot0.t = t
wdot_old.sub(0).interpolate(udot0.eval)
thetadot0 = thetadot_exact()
thetadot0.t = t
wdot_old.sub(1).interpolate(thetadot0.eval)

w.sub(0).name = 'u_mid'
w.sub(1).name = 'theta'

u_exact_func = Function(W.sub(0).collapse()[0])
u_exact_i = u_exact()

def wdot_vector(w, w_old, wdot_old):
    return 2/dt*w.vector - 2/dt*w_old.vector - wdot_old.vector

####### Time stepping #################
for i in range(0,Nsteps):

    t += dt
    print("------- Time step "+str(i+1)+"/"+str(Nsteps)
            +" , t = "+str(t)+" -------")

    # Solve the nonlinear problem for this time step and put the solution
    # (in homogeneous coordinates) in y_hom.
    solveNonlinear(F,w,bcs,log=False)

    # Advance to the next time step
    # ** since u_dot, theta_dot are not functions, we cannot directly 
    # ** interpolate them onto wdot_old.
    wdot_old.vector[:] = wdot_vector(w,w_old,wdot_old)
    w_old.interpolate(w)

    # Save solution to XDMF format
    xdmf_file.write_function(w.sub(0), t)
    xdmf_file_rotation.write_function(w.sub(1), t)
    u_exact_i.t = t
    u_exact_func.interpolate(u_exact_i.eval)
    xdmf_file_exact.write_function(u_exact_func, t)

    # Record the displacement
    u_output[i+1] = w.sub(0).eval(coord, cell_id)[2]
    L2_error[i] = sqrt(assemble_scalar(form((u - u_exact_func)**2*dx)))

########## Outputs: ##############
u_output_max = np.full((Nsteps+1), np.max(u_output))
u_output_exact_max = np.full((Nsteps+1), np.max(u_output_exact))

print("="*40)
print("="*10+" Mode "+str(r)+" at x = "+"1/"+str(2**r)+"L "+"="*10)
print("="*40)

print("Max u_output:", np.max(abs(u_output)))
print("u_output = 0 at t =", time[np.where(np.abs(u_output) < 1e-2)])
print("Max u_output_exact:", np.max(u_output_exact))
print("u_output_exact = 0 at t =", time[np.where(np.abs(u_output_exact) < 1e-2)])

########## Visualization: ##############
from matplotlib import pyplot as plt
# Plot displacement evolution at the location of interest
plt.figure()
plt.plot(time, u_output)
plt.plot(time, u_output_exact, '--')
plt.plot(time, np.zeros(len(time)), '--', color='grey')
plt.plot(time, u_output_max, '--', color='grey')
plt.plot(time, u_output_exact_max, '--', color='grey')
plt.xlabel("Time")
plt.ylabel("Vertical displacement")
plt.ylim(-2*C, 2*C)
plt.legend(['Numerical solution', 'Exact solution'])
plt.savefig(PATH+"mode_"+str(r)+"_displacement.png")

# Plot L2 error for the displacement field wrt time
plt.figure()
plt.plot(time, L2_error)
plt.xlabel("Time")
plt.ylabel("error")
plt.savefig(PATH+"mode_"+str(r)+"_L2_error.png")
