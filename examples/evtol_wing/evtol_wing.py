"""
Structural analysis on an eVTOL wing model
Boeing 777-9x Specifications: (units: m/ft, kg/lb)
(from https://en.wikipedia.org/wiki/Boeing_777X)
Maximum take-off weight	352,400kg/777,000lb
Wing span (extended)    71.75m/235.42ft
Overall length	        76.73m/251.75ft
"""
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *

tri_mesh = [
            "eVTOL_wing_half_tri_77020_103680.xdmf", # error
            "eVTOL_wing_half_tri_81475_109456.xdmf", # error
            "eVTOL_wing_half_tri_107695_136686.xdmf",
            "eVTOL_wing_half_tri_135957_170304.xdmf"] # error

quad_mesh = [
            "eVTOL_wing_half_quad_77020_51840.xdmf", # error
            "eVTOL_wing_half_quad_81475_54228.xdmf", # error
            "eVTOL_wing_half_quad_107695_68343.xdmf", # error
            "eVTOL_wing_half_quad_135957_85152.xdmf",] # error
test = 1
# file_name = quad_mesh[test]
file_name = tri_mesh[test]
mesh_file = "../../mesh/mesh-examples/evtol-wing/" + file_name
with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
# sometimes it should be `name="mesh"` to avoid the error

E_val = 6.8E10 # unit: Pa (N/m^2)
nu_val = 0.35
h_val = 3E-3 # overall thickness (unit: m)

# Scaled body force
f_d = 40254*h_val # force per unit area (unit: N/m^2)

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f = as_vector([0,0,f_d]) # Body force per unit area

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

############ Set the BCs for the airplane model ###################

u0 = Function(W)
u0.vector.set(0.0)


locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]

########## Solve with Newton solver wrapper: ##########
solveNonlinear(F,w,bcs)

########## Output: ##############

u_mid, _ = w.split()

dofs = len(w.vector.getArray())

uZ = computeNodalDisp(w.sub(0))[2]
print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(uZ))
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", dofs)
print("-"*50)


with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid_tri_"+str(dofs)+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
