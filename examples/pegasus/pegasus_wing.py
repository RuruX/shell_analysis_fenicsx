"""
Structural analysis on a Pegasus wing model
-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import assemble_scalar
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *
from shell_analysis_fenicsx.read_properties import readCLT, sortIndex


file_name = "pegasus_6257_quad_SI.xdmf"
path = "../../mesh/mesh-examples/pegasus/mesh_from_michael_SI/"
mesh_file = path + file_name
with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
# sometimes it should be `name="mesh"` to avoid the error
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

E_val = 6.8E10 # unit: Pa (N/m^2)
nu_val = 0.35
h_val = 3E-3 # overall thickness (unit: m)

# Scaled body force
f_d = 10. # force per unit area (unit: N/m^2)

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio

################### Constant thickness  ###################
# h = Constant(mesh,h_val) # Shell thickness

################### Varying thickness distribution ###################
hrcs = np.reshape(np.loadtxt(path+'pegasus_t_med_SI.csv'),(nn,1))
h_nodal = np.arange(nn)
node_indices = mesh.geometry.input_global_indices
h_array = sortIndex(hrcs, h_nodal, node_indices)
############# Apply element-wise thickness ###################
VT = FunctionSpace(mesh, ("CG", 1))
h = Function(VT)
h.vector.setArray(h_array)

################## Aerodynamic loads ###################
# Uniform loads
f = as_vector([0,0,f_d]) # Body force per unit area

# ############### Read and apply nodal force #################
# # Element-wise loads
# frcs = np.reshape(np.loadtxt(path+'aero_force_test.txt'),(nn,3))
# f_nodes = np.arange(nn)
# # map input nodal indices to dolfinx index structure
# coords = mesh.geometry.x
# node_indices = mesh.geometry.input_global_indices
# f_array = sortIndex(frcs, f_nodes, node_indices)

# # apply array in function space
# VF = VectorFunctionSpace(mesh, ("CG", 1))
# f = Function(VF)
# f.vector.setArray(f_array) # Body force per unit area
# ###############################################################

element_type = "CG2CG1"
#element_type = "CG2CR1"


element = ShellElement(
                mesh,
                element_type,
#                inplane_deg=3,
#                shear_deg=3
                )

# VE1 = VectorElement("Lagrange",mesh.ufl_cell(),1)
# WE = MixedElement([VE1,VE1])
# W = FunctionSpace(mesh,WE)

W = element.W
w = Function(W)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)

############ Set the BCs for the airplane model ###################

u0 = Function(W)
u0.vector.set(0.0)


locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.less(x[1], 1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.less(x[1], 1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]
########## Solve with Newton solver wrapper: ##########
from timeit import default_timer
start = default_timer()
solveNonlinear(F,w,bcs,log=True)
stop = default_timer()
print("Time for solve nonlinear:", stop-start)
########## Output: ##############

u_mid, _ = w.split()

dofs = len(w.vector.getArray())

uZ = computeNodalDisp(w.sub(0))[2]
# x_tip = [9.77099,12.2157,1.06831]
# cell_tip = 6402
# uZ_tip = u_mid.eval(x_tip, cell_tip)[-1]
strain_energy = assemble_scalar(form(elastic_energy))
print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(uZ))
# print("Tip deflection:", uZ_tip)
print("Total strain energy:", strain_energy)
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", dofs)
print("-"*50)

with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid_tri_"+str(dofs)+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
with XDMFFile(MPI.COMM_WORLD, "solutions/thickness_nodal_"+str(dofs)+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(h)

shell_stress_RM = ShellStressRM(mesh, w, h, E, nu)
von_Mises_top = shell_stress_RM.vonMisesStress(h/2)
V1 = FunctionSpace(mesh, ('CG', 1))
von_Mises_top_func = Function(V1)
project(von_Mises_top, von_Mises_top_func, lump_mass=True)

with XDMFFile(MPI.COMM_WORLD, "solutions/von_Mises_top"+str(dofs)+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(von_Mises_top_func)
