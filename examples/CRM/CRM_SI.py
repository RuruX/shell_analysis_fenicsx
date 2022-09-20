"""
Structural analysis for the common research model (CRM)
-----------------------------------------------------------
Run the test with different meshes by

`python3 CRM_SI.py --eleSize YOUR_CHOICE_OF_SIZE`

where `YOUR_CHOICE_OF_SIZE` should be an interger within [1,11]
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *
# (interactive visualization not available if using docker container)
# from shell_analysis_fenicsx.pyvista_plotter import plotter_3d


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eleSize',dest='eleSize',default=11,
                    help='Element size.')

args = parser.parse_args()
element_size = int(args.eleSize)

tri_mesh = ["crm_metallic_structure_tri_SI_"+str(element_size)+"in.xdmf"]

quad_mesh = []

test = 0
#file_name = quad_mesh[test]
file_name = tri_mesh[test]
mesh_file = "../../mesh/mesh-examples/CRM/" + file_name
with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
# sometimes it should be `name="mesh"` to avoid the error



## original units: imperial
#E_val = 1.03E7 # unit: psi
#nu_val = 0.33
#h_val = 0.625 # overall thickness (unit: inch)
#rho = 0.0975 # unit: lb/inch^3
## Scaled body force
#gravity = 386.088582677165  # unit: in/m^2
#f_d = -1.0*rho*gravity*h_val # force per unit area

# units: SI
E_val = 7.1016E10 # unit: pa
nu_val = 0.33
h_val = 0.015875 # overall thickness (unit: m)
rho = 2698.79071 # unit: kg/m^3
gravity = 9.807 # unit: m/s^2 (N/kg)
# Scaled body force
f_d = -1.0*rho*gravity*h_val # force per unit area


f = ufl.as_vector([0,0,f_d])
element_type = "CG2CG1"
#element_type = "CG2CR1"


element = ShellElement(
                mesh,
                element_type,
                inplane_deg=3,
                shear_deg=3
                )
# W = element.W
cell = mesh.ufl_cell()
VE1 = VectorElement("Lagrange",cell,1)
VE2 = VectorElement("Lagrange",cell,1)
WE = MixedElement([VE2,VE1])
W = FunctionSpace(mesh,WE)
w = Function(W)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E_val,nu=nu_val,h=h_val)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E_val, h_val, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)

############ Set the BCs for the airplane model ###################

u0 = Function(W)
u0.vector.set(0.0)

# Define BCs geometrically
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
solveNonlinear(F,w,bcs,max_it=3)

########## Output: ##############

u_mid, _ = w.split()

with XDMFFile(MPI.COMM_WORLD, "solutions/displacement_"+str(element_size)+"in.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)

uZ = computeNodalDisp(w.sub(0))[2]

vertex_values = uZ

print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
#print("Tip deflection:", vertex_values[11]) # node 0 is where the maximum is
print("Maximum z-deflection at the tip:", -np.max(abs(uZ)))
print("Strain energy:", assemble_scalar(form(elastic_energy)))
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", len(w.vector.getArray()))
print("-"*50)

####### Project the CG2CG1 space onto CG1CG1 for plotting ##############


# plotter = plotter_3d(mesh, vertex_values)
# plotter.show()
