"""
Structural analysis of the classic shell obstacle course:
1/3: Scordelis-Lo Roof
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *


roof = [#### tri mesh ####
        "roof_tri_30_20.xdmf",
        "roof_tri_60_40.xdmf",
        "roof5_25882.xdmf",
        "roof6_104524.xdmf",
        "roof12_106836.xdmf",
        #### quad mesh ####
        "roof_quad_3_2.xdmf",
        "roof_quad_6_4.xdmf",
        "roof_quad_12_8.xdmf",
        "roof_quad_24_16.xdmf",
        "roof_quad_30_20.xdmf",
        "roof_quad_60_40.xdmf",
        "roof_quad_120_80.xdmf",
        "roof_quad_240_160.xdmf",
        "roof_quad_360_240.xdmf"]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_id',dest='meshID',default=0,
                    help='mesh id.')

args = parser.parse_args()
mesh_id = int(args.meshID)


filename = "../../mesh/mesh-examples/scordelis-lo-roof/"+roof[mesh_id]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

num_el = mesh.topology.index_map(mesh.topology.dim).size_local
num_vertice = mesh.topology.index_map(0).size_local
E_val = 4.32e8
nu_val = 0.0
h_val = 0.25
f_d = -90.

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
# h = Constant(mesh,h_val) # Shell thickness
# f = ufl.as_vector([0,0,f_d]) # Body force per unit area
############ Apply element-wise thickness ###############
h_array = np.full((num_el,),h_val)
VT = FunctionSpace(mesh, ("DG", 0))
h = Function(VT)
h.vector.setArray(h_array) # Body force per unit area

############# Apply distributed loads ###################
f_array = np.tile([0,0,f_d], num_vertice)
VF = VectorFunctionSpace(mesh, ("CG", 1))
f = Function(VF)
f.vector.setArray(f_array) # Body force per unit area

element_type = "CG2CG1" # with quad/tri elements
#element_type = "CG2CR1" # with tri elements

element = ShellElement(
                mesh,
                element_type,
#                inplane_deg=3,
#                shear_deg=3
                )
W = element.W
w = Function(W)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


# # =====================================================
# # Compute the CLT matrices by CSDL model
# from csdl_om import Simulator
# G_val = E_val/2/(1+nu_val)
# mat_prop = np.array([E_val,E_val,nu_val,G_val,nu_val]) #isotropic material
# no_plies = 1
# ply_stack = np.radians(np.array([90]))
# sim = Simulator(CLT(no_plies=no_plies, mat_prop=mat_prop, ply_stack=ply_stack, h0=h_val))
# sim.run()
# A_local = sim['A'].astype('float64').flatten()
# B_local = sim['B'].astype('float64').flatten()
# D_local = sim['D'].astype('float64').flatten()
# A_s_local = sim['A_star'].astype('float64').flatten()
# A_array = np.tile(A_local,num_el)
# B_array = np.tile(B_local,num_el)
# D_array = np.tile(D_local,num_el)
# A_s_array = np.tile(A_s_local,num_el)
# CLT_data = (A_array, B_array, D_array, A_s_array)
# # =====================================================

# #### Compute the CLT model from the material properties (for single-layer material)
# material_model = MaterialModelComposite(mesh,CLT_data)
# elastic_model = ElasticModel(mesh,w,material_model.CLT)
# elastic_energy = elastic_model.elasticEnergy(dx_inplane=dx_inplane,
#                                             dx_shear=dx_shear)
# F = elastic_model.weakFormResidual(elastic_energy, f)

#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h,BOT=True) # Simple isotropic material
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)

############ Set the BCs for the Scordelis-Lo roof problem ###################
ubc = Function(W)
ubc.vector.set(0.0)

locate_BC1 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC3 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC4 = locate_dofs_geometrical((W.sub(1).sub(0), W.sub(1).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC5 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC6 = locate_dofs_geometrical((W.sub(0).sub(0), W.sub(0).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC7 = locate_dofs_geometrical((W.sub(1).sub(1), W.sub(1).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC8 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC2, W.sub(0).sub(2)),
       dirichletbc(ubc, locate_BC3, W.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC4, W.sub(1).sub(0)),
       dirichletbc(ubc, locate_BC5, W.sub(1).sub(2)),
       dirichletbc(ubc, locate_BC6, W.sub(0).sub(0)),
       dirichletbc(ubc, locate_BC7, W.sub(1).sub(1)),
       dirichletbc(ubc, locate_BC8, W.sub(1).sub(2))
       ]

########## Solve with Newton solver wrapper: ##########
solveNonlinear(F, w, bcs,log=True)

########## Output: ##############
num_el = mesh.topology.index_map(mesh.topology.dim).size_local
num_vertice = mesh.topology.index_map(0).size_local
num_dof = len(w.vector.getArray())
uZ = computeNodalDisp(w.sub(0))[2]
# Comparing the results to the numerical solution
print("Scordelis-Lo roof theory tip deflection: v_tip = -0.3018")
print("Tip deflection:", min(uZ))
print("  Number of elements = "+str(num_el))
print("  Number of vertices = "+str(num_vertice))
print("  Number of DoFs = "+str(num_dof))
########## Visualization: ##############

u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)

File = open('results.txt', 'a')
outputs = '\n{0:5d}      {1:5d}      {2:6d}     {3:.6f}'.format(
            num_el, num_vertice, num_dof, min(uZ))
File.write(outputs)
File.close()
