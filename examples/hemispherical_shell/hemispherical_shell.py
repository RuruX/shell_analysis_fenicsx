"""
Structural analysis of the classic shell obstacle course:
3/3: hemispherical shell
-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *


path = "../../mesh/mesh-examples/hemispherical-shell/"
# mesh refinements (4, 8, 16 and 32 elements per edge)
sphere = [#### tri mesh ####
        "sphere_tri_4_4.xdmf",
        "sphere_tri_8_8.xdmf",
        "sphere_tri_16_16.xdmf",
        "sphere_tri_32_32.xdmf",
        "sphere_tri_64_64.xdmf",
        "sphere_tri_128_128.xdmf",
        ### quad mesh ####
        "sphere_quad_4_4.xdmf",
        "sphere_quad_8_8.xdmf",
        "sphere_quad_16_16.xdmf",
        "sphere_quad_32_32.xdmf",
        "sphere_quad_64_64.xdmf",
        "sphere_quad_128_128.xdmf",
        "sphere_quad_deg_6_64_64.xdmf",
        "sphere_quad_deg_30_64_64.xdmf",
        ]


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_id',dest='meshID',default=0,
                    help='mesh id.')

args = parser.parse_args()
mesh_id = int(args.meshID)


filename = sphere[mesh_id]
meshfile = path + filename
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshfile, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

radius = 10.0
E_val = 6.825e+7
nu_val = 0.3
h_val = 0.04
f_x_val = 2.0/2
f_y_val = -2.0/2

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f_x = ufl.as_vector([f_x_val,0.,0.]) # Point load in x-direction
f_y = ufl.as_vector([0.,f_y_val,0.]) # Point load in x-direction
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


#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f_x+f_y)

############ Set the BCs for the Scordelis-Lo roof problem ###################
u0 = Function(W)
u0.vector.set(0.0)


locate_BC1 = locate_dofs_geometrical((W.sub(0).sub(1), W.sub(0).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0., atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1).sub(0), W.sub(1).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0., atol=1e-6))
locate_BC3 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0., atol=1e-6))

locate_BC4 = locate_dofs_geometrical((W.sub(0).sub(0), W.sub(0).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0., atol=1e-6))
locate_BC5 = locate_dofs_geometrical((W.sub(1).sub(1), W.sub(1).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0., atol=1e-6))
locate_BC6 = locate_dofs_geometrical((W.sub(1).sub(2), W.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0., atol=1e-6))


ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC2, W.sub(1).sub(0)),
       dirichletbc(ubc, locate_BC3, W.sub(1).sub(2)),
       dirichletbc(ubc, locate_BC4, W.sub(0).sub(0)),
       dirichletbc(ubc, locate_BC5, W.sub(1).sub(1)),
       dirichletbc(ubc, locate_BC6, W.sub(1).sub(2)),
       ]

########### Apply the point load #############################


delta_x = Delta(x0=np.array([radius, 0.0, 0.0]), f_p=(f_x_val, 0., 0.))
delta_y = Delta(x0=np.array([0.0, radius, 0.0]), f_p=(0., f_y_val, 0.))

f1 = Function(W)
f1_0,_ = f1.split()
f1_0.interpolate(delta_x.eval)

f2 = Function(W)
f2_0,_ = f2.split()
f2_0.interpolate(delta_y.eval)

# Assemble linear system
a = derivative(F,w)
L = -F
A = assemble_matrix(form(a), bcs)
A.assemble()
b = assemble_vector(form(L))
b.setArray(f1.vector.getArray()+f2.vector.getArray())
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)



######### Solve the linear system with KSP solver ############
solveKSP(A, b, w.vector)

########## Output: ##############
num_el = mesh.topology.index_map(mesh.topology.dim).size_local
num_vertice = mesh.topology.index_map(0).size_local
num_dof = len(w.vector.getArray())
# Comparing the results to the numerical solution
uX, uY, uZ = computeNodalDisp(w.sub(0))
print("-------- Hemispherical Shell Analysis -----------")
print("---------- with "+filename+" ----------")
print("----- (radial deflection at the point load) -----")
print("Reference solution: u_ref = ", 0.09355, "(with 18 degree cutout)")
print("Shell analysis solution:", max(uX))
print("  Number of elements = "+str(num_el))
print("  Number of vertices = "+str(num_vertice))
print("  Number of DoFs = "+str(num_dof))
u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
File = open('results.txt', 'a')
outputs = '\n{0:5d}      {1:5d}      {2:6d}     {3:.6f}'.format(
            num_el, num_vertice, num_dof, max(uX))
File.write(outputs)
File.close()
