"""
Structural analysis of the classic cantilever beam problem

-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *
# (interactive visualization not available if using docker container)
# from shell_analysis_fenicsx.pyvista_plotter import plotter_3d


beam = [#### quad mesh ####
        "plate_2_10_quad_1_5.xdmf",
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",
        "plate_2_10_quad_20_100.xdmf",
        "plate_2_10_quad_40_200.xdmf",
        "plate_2_10_quad_80_400.xdmf",]

filename = "../../mesh/mesh-examples/clamped-RM-plate/"+beam[3]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

E_val = 4.32e8
nu_val = 0.0
h_val = 0.2
width = 2.
length = 10.
f_d = 10.*h_val

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f = ufl.as_vector([0,0,f_d]) # Body force per unit surface area

element_type = "CG2CG1"
#element_type = "CG2CR1"

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
F = elastic_model.weakFormResidual(elastic_energy, f)

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

########### Apply the point load #############################
x_pt = 3.
f_val = 10.
r_x_pt = length - x_pt

x_pt_1 = 8.
f_val_1 = 10.
r_x_pt_1 = length - x_pt

delta = Delta_mpt(x0=np.array([[x_pt, 1.0, -1.0],[x_pt_1, 1.0, -1.0]]),
                f_p=np.array([[0.,0.,f_val],[0.,0.,f_val_1]]))
f1 = Function(W)
f1_0,_ = f1.split()
print("call evaluation...")
f1_0.interpolate(delta.eval)


f1_x = computeNodalDisp(f1.sub(0))[0]
f1_y = computeNodalDisp(f1.sub(0))[1]
f1_z = computeNodalDisp(f1.sub(0))[2]
print("-"*60)
print("                               Projected CG2   "+"     Original     ")
print("-"*60)
print("Sum of forces in x-direction:", np.sum(f1_x))
print("Sum of forces in y-direction:", np.sum(f1_y))
print("Sum of forces in z-direction:", np.sum(f1_z))
print("-"*60)
# Assemble linear system
a = derivative(F,w)
L = -F
A = assemble_matrix(form(a), bcs)
A.assemble()
b = assemble_vector(form(L))

b.setArray(f1.vector.getArray())
# b.setArray(f1.vector.getArray()+f2.vector.getArray())
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, bcs)



######### Solve the linear system with KSP solver ############
solveKSP(A, b, w.vector)
# ########## Solve with Newton solver wrapper: ##########
# solveNonlinear(F,w,bcs)

# Comparing the solution to the Kirchhoff analytical solution
uZ = computeNodalDisp(w.sub(0))[2]
Ix = width*h_val**3/12
print(Ix,f_val,x_pt,E_val,r_x_pt)
########## Output: ##########
def EB_deflection(f_val,x_pt):
    return (f_val*x_pt**2/(6*E_val*Ix))*(3*length-x_pt)
print("Euler-Beinoulli Beam theory deflection:",
    EB_deflection(f_val,x_pt)+EB_deflection(f_val_1,x_pt_1))
print("Reissner-Mindlin FE deflection:", max(abs(uZ)))

print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))

########## Visualization: ##############

u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
# with XDMFFile(MPI.COMM_WORLD, "solutions/distributed_point_loads.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(f_1)
#### (interactive visualization not available if using docker container)
# vertex_values = uZ
# plotter = plotter_3d(mesh, vertex_values)
# plotter.show()
