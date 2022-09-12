"""
Structural analysis of the classic shell obstacle course:
2/3: Pinched Cylinder
"""

from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsX import *
# (interactive visualization not available if using docker container)
# from shell_analysis_fenicsX.pyvista_plotter import plotter_3d


path = "../../mesh/mesh-examples/pinched-cylinder/"
# mesh refinements (4, 8, 16 and 32 elements per edge)
cylinder = [#### tri mesh ####
        "cylinder_tri_4_4.xdmf", 
        "cylinder_tri_8_8.xdmf", 
        "cylinder_tri_16_16.xdmf", 
        "cylinder_tri_32_32.xdmf", 
        "cylinder_tri_64_64.xdmf", 
        "cylinder_tri_128_128.xdmf", 
        ### quad mesh ####
        "cylinder_quad_4_4.xdmf",
        "cylinder_quad_8_8.xdmf",
        "cylinder_quad_16_16.xdmf",
        "cylinder_quad_32_32.xdmf",
        "cylinder_quad_64_64.xdmf",
        "cylinder_quad_128_128.xdmf",
        ]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_id',dest='meshID',default=0,
                    help='mesh id.')

args = parser.parse_args()
mesh_id = int(args.meshID)


filename = cylinder[mesh_id]
meshfile = path + filename
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshfile, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

length = 600.
radius = 300.
E_val = 3.0e6
nu_val = 0.3
h_val = 3.0
f_val = -1.0/4

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f = ufl.as_vector([0.,0.,f_val]) # Point-force in z-direction

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


# Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)

############ Set the BCs for the Scordelis-Lo roof problem ###################
ubc = Function(W)
ubc.vector.set(0.0)


locate_BC1 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(2).collapse()[0]), 
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC3 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]), 
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC4 = locate_dofs_geometrical((W.sub(1).sub(0), W.sub(1).sub(0).collapse()[0]),    
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC5 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]), 
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC6 = locate_dofs_geometrical((W.sub(0).sub(0), W.sub(0).sub(0).collapse()[0]), 
                                    lambda x: np.isclose(x[0], length/2. ,atol=1e-6))
locate_BC7 = locate_dofs_geometrical((W.sub(1).sub(1), W.sub(1).sub(1).collapse()[0]), 
                                    lambda x: np.isclose(x[0], length/2. ,atol=1e-6))
locate_BC8 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]), 
                                    lambda x: np.isclose(x[0], length/2. ,atol=1e-6))
locate_BC9 = locate_dofs_geometrical((W.sub(0).sub(2), W.sub(0).sub(1).collapse()[0]), 
                                    lambda x: np.isclose(x[2], 0. ,atol=1e-6))
locate_BC10 = locate_dofs_geometrical((W.sub(1).sub(0), W.sub(1).sub(0).collapse()[0]),    
                                    lambda x: np.isclose(x[2], 0. ,atol=1e-6))
locate_BC11 = locate_dofs_geometrical((W.sub(1).sub(1), W.sub(1).sub(2).collapse()[0]), 
                                    lambda x: np.isclose(x[2], 0. ,atol=1e-6))

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC2, W.sub(0).sub(2)),
       dirichletbc(ubc, locate_BC3, W.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC4, W.sub(1).sub(0)),
       dirichletbc(ubc, locate_BC5, W.sub(1).sub(2)),
       dirichletbc(ubc, locate_BC6, W.sub(0).sub(0)),
       dirichletbc(ubc, locate_BC7, W.sub(1).sub(1)),
       dirichletbc(ubc, locate_BC8, W.sub(1).sub(2)),
       dirichletbc(ubc, locate_BC9, W.sub(0).sub(2)),
       dirichletbc(ubc, locate_BC10, W.sub(1).sub(0)),
       dirichletbc(ubc, locate_BC11, W.sub(1).sub(1))
       ]
       
########### Apply the point load #############################


delta = Delta(x0=np.array([length/2., 0.0, radius]), f_p=(0.,0.,f_val))

f1 = Function(W)
f1_0,_ = f1.split()
f1_0.interpolate(delta.eval)


# Assemble linear system
a = derivative(F,w)
L = -F
A = assemble_matrix(form(a), bcs)
A.assemble()
b = assemble_vector(form(L))
b.setArray(f1.vector.getArray())
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)



######### Solve the linear system with KSP solver ############
solveKSP(A, b, w.vector)

########## Output: ##############

u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
num_el = mesh.topology.index_map(mesh.topology.dim).size_local
num_vertice = mesh.topology.index_map(0).size_local
num_dof = len(w.vector.getArray())
# Comparing the results to the numerical solution
magnitude = computeNodalDispMagnitude(w.sub(0))
print("----------- Pinched cylinder solution -----------")
print("---------- with "+filename+" ----------")
print("----- (radial deflection at the point load) -----")
print("Reference solution: u_ref = ", 1.8248e-5)
print("Shell analysis solution:", max(magnitude))
print("  Number of elements = "+str(num_el))
print("  Number of vertices = "+str(num_vertice))
print("  Number of DoFs = "+str(num_dof))
# # Visualization with Pyvista (not available if using docker container)
# vertex_values = magnitude
# plotter = plotter_3d(mesh, vertex_values)
# plotter.show()
File = open('results.txt', 'a')
outputs = '\n{0:5d}      {1:5d}      {2:6d}     {3:.5e}'.format(
            num_el, num_vertice, num_dof, max(magnitude))
File.write(outputs)
File.close()