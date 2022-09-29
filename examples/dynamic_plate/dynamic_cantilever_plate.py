"""
Dynamic structural analysis of the cantilever beam released from deformation

-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""
from logging import WARNING
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *


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
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# E_val = 4.32e8
# nu_val = 0.0
E_val = 1e7 # unit: pa
nu_val = 0.3
h_val = 0.2
width = 2.
length = 10.

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
rho = Constant(mesh,1.0)
# Time-stepping parameters
T       = 5.0
Nsteps  = 50
dt = T/Nsteps
p0 = 1.
cutoff_Tc = T/5

f_0 = Constant(mesh, (0.0,0.0,0.0))
def f(t):
    f_val = 0.0
    # Added some spatial variation here. Expression is sin(t)*x
    if t <= cutoff_Tc:
        f_val = p0*t/cutoff_Tc
    return Constant(mesh, (0.0,0.0,f_val))

element_type = "CG2CG1"
element = ShellElement(
                mesh,
                element_type,
                )
W = element.W
w = Function(W)
u, theta = split(w)
# Quantities from the previous time step
w_old = Function(W)
wdot_old = Function(W)
u_old, theta_old = split(w_old)
udot_old, thetadot_old = split(wdot_old)


u_mid = 0.5*(u_old+u)
theta_mid = 0.5*(theta_old+theta)

udot = Constant(mesh, 2/dt)*u - Constant(mesh, 2/dt)*u_old - udot_old
uddot = (udot - udot_old)/dt
thetadot = Constant(mesh, 2/dt)*theta - Constant(mesh, 2/dt)*theta_old - thetadot_old
thetaddot = (thetadot - thetadot_old)/dt
wdot = Constant(mesh, 2/dt)*w - Constant(mesh, 2/dt)*w_old - wdot_old
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear

#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = DynamicElasticModel(mesh, u_mid, theta_mid, material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)

dw = TestFunction(W)
du_mid,dtheta = split(dw)

dWint = elastic_model.weakFormResidual(elastic_energy, w, dw, f_0)

# Inertial contribution to the residual:
dWmass = elastic_model.inertialResidual(rho,h,uddot,thetaddot)

######### Set the BCs to have all the dofs equal to 0 on the left edge ##########
# Define BCs geometrically
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]

# Create a time integrator for the displacement.
# timeInt = BackwardEulerIntegrator(mesh, dt, w,
#                                      (w_old, wdot_old))
# timeInt = ImplicitMidpointIntegrator(mesh, dt, w,
#                                      (w_old, wdot_old))
# Time-stepping
time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
f_t = np.zeros((Nsteps+1,))
for i in range(0,round(cutoff_Tc/dt)+1):
    f_t[i] = p0*i*dt/cutoff_Tc
xdmf_file = XDMFFile(MPI.COMM_WORLD, "solutions/displacement.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file_stress = XDMFFile(MPI.COMM_WORLD, "solutions/stress.xdmf", "w")
xdmf_file_stress.write_mesh(mesh)
t = 0.

# Define function space for stresses
Vsig = TensorFunctionSpace(mesh, ("DG", 0), shape=(2,2))
sig = Function(Vsig, name="sigma")

w.sub(0).name = 'u_mid'
sig.name = 'membrane stress'

for i in range(0,Nsteps):

    t += dt
    print("------- Time step "+str(i+1)
            +" , t = "+str(t)+" -------")
    dWext = -inner(f(t),du_mid)*dx
    F = dWmass + dWint + dWext
    # Solve the nonlinear problem for this time step and put the solution
    # (in homogeneous coordinates) in y_hom.
    solveNonlinear(F,w,bcs)
    # Advance to the next time step.
    # timeInt.advance()

    project(wdot, wdot_old)
    w_old.interpolate(w)
    # Save solution to XDMF format
    xdmf_file.write_function(w.sub(0), t)

    # Compute stresses and save to file
    sigma = elastic_model.plane_stress_elasticity(E, nu)
    project(sigma, sig)
    xdmf_file_stress.write_function(sig, t)

    # Record tip displacement and compute energies
    # u_tip[i+1] = evaluateFunc(w.sub(0), [10., 1.0, 0.], mesh)[2]
    u_tip[i+1] = w.sub(0).eval([[10., 1.0, 0.]], [74])[2]

########## Visualization: ##############

from matplotlib import pyplot as plt
# Plot tip displacement evolution
plt.figure()
plt.plot(time, u_tip)
plt.xlabel("Time")
plt.ylabel("Tip displacement")
plt.ylim(-0.8, 0.8)
plt.savefig("solutions/tip_displacement.png")

# Plot load evolution
plt.figure()
plt.plot(time, f_t)
plt.xlabel("Time")
plt.ylabel("Load")
plt.ylim(-0.2, 1.2)
plt.savefig("solutions/Load.png")
