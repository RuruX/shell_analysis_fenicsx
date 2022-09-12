"""
Structural analysis for the undeflected common research model (uCRM)
uCRM-9 Specifications: (units: m/ft, kg/lb)
(from https://deepblue.lib.umich.edu/bitstream/handle/2027.42/143039/6.2017-4456.pdf?sequence=1)
Maximum take-off weight	352,400kg/777,000lb
Wing span (extended)    71.75m/235.42ft
Overall length	        76.73m/251.75ft
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import locate_dofs_topological, Constant, dirichletbc
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsX import *

########## Reference results with ICEM meshes ######
############### unit: meter ########################
# --------------------------------------------------
# -------- uCRM-9_coarse.xdmf ---------
# --------------------------------------------------
#   Tip deflection: 0.17673840967897833
#   Number of elements = 25055
#   Number of vertices = 23738
# --------------------------------------------------
# -------- uCRM-9_medium.xdmf ---------
# --------------------------------------------------
#   Tip deflection: 0.1778104969452804
#   Number of elements = 61451
#   Number of vertices = 59388
# --------------------------------------------------
# -------- uCRM-9_fine.xdmf ---------
# --------------------------------------------------
#   Tip deflection: 0.1780523716805102
#   Number of elements = 97920
#   Number of vertices = 95286


# ### mesh 5 ####
# shellmesh = ["uCRM_shellmesh_2368_quad_2802.xdmf", #Tip deflection: 0.17520083913934492
#             "uCRM_shellmesh_3442_quad_3962.xdmf", #Tip deflection: 0.17526195416951484
#             "uCRM_shellmesh_10294_quad_11208.xdmf", #Tip deflection: 0.1758998220448985
#             "uCRM_shellmesh_14762_quad_15848.xdmf", #Tip deflection: 0.17598339341893093
#             "uCRM_shellmesh_42958_quad_44832.xdmf", #Tip deflection: 0.17702357582506908
#             "uCRM_shellmesh_61174_quad_63392.xdmf", #Tip deflection: 0.17708406474376345
#             "uCRM_shellmesh_175534_quad_179328.xdmf", #Tip deflection: 0.1775895933245462
#             "uCRM_shellmesh_249086_quad_253568.xdmf"] #Tip deflection:

## mesh 6 ####

# PATH_SHELLMESH = "../../mesh/mesh-examples/uCRM-9-ShellMesh/mesh_6/"
# shellmesh = ["uCRM_shellmesh_2362_quad_2798.xdmf", #Tip deflection: 0.1752159718463274
#             "uCRM_shellmesh_10270_quad_11192.xdmf", #Tip deflection: 0.17581703066097043
#             "uCRM_shellmesh_14814_quad_15928.xdmf", #Tip deflection: 0.1758840767800547
#             "uCRM_shellmesh_42824_quad_44768.xdmf", #Tip deflection: 0.1768241018497809
#             "uCRM_shellmesh_61434_quad_63712.xdmf", #Tip deflection: 0.17688297983022433
#             "uCRM_shellmesh_175234_quad_179072.xdmf",] #Tip deflection: 0.17738536556378479

## mesh 7 ####
PATH_SHELLMESH = "../../mesh/mesh-examples/uCRM-9-ShellMesh/mesh_7/"
shellmesh = ["uCRM_shellmesh_2364_quad_2800.xdmf", #Tip deflection:
            "uCRM_shellmesh_3456_quad_3988.xdmf", #Tip deflection:
            "uCRM_shellmesh_10278_quad_11200.xdmf", #Tip deflection:
            "uCRM_shellmesh_14838_quad_15952.xdmf", #Tip deflection:
            "uCRM_shellmesh_42906_quad_44800.xdmf", #Tip deflection:
            "uCRM_shellmesh_61530_quad_63808.xdmf", #Tip deflection:
            "uCRM_shellmesh_175362_quad_179200.xdmf",] #Tip deflection: 

PATH_ICEM = "../../mesh/mesh-examples/uCRM-9-ICEM/"
ICEM_mesh = ["uCRM-9_coarse.xdmf", #Tip deflection: 0.17673840967897833
            "uCRM-9_medium.xdmf", #Tip deflection: 0.1778104969452804
            "uCRM-9_fine.xdmf"] #Tip deflection: 0.1780523716805102

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mesh_id',dest='meshID',default=0,
                    help='mesh id.')

args = parser.parse_args()
mesh_id = int(args.meshID)


# file_name = shellmesh[mesh_id]
# PATH = PATH_SHELLMESH

# mesh_file = PATH + file_name
# with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
#        mesh = xdmf.read_mesh(name="Grid")
# sometimes it should be `name="mesh"` to avoid the error

file_name = ICEM_mesh[mesh_id]
PATH = PATH_ICEM
mesh_file = PATH + file_name
with XDMFFile(MPI.COMM_WORLD, mesh_file, "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
       mesh = xdmf.read_mesh(name="Grid")

def boundary(x):
    return np.logical_or(x[1] < 3.1, x[1] > 30.0)

wing_area = calculateSurfaceArea(mesh, boundary)

E_val = 7.31E10 # unit: (N/m^2)
nu_val = 0.3
h_val = 3E-3 # overall thickness (unit: m)

# Scaled body force
f_d = 2780*9.81*h_val # force per unit area (unit: N/m^2)

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f = Constant(mesh,(0,0,f_d)) # Body force per unit area

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


facets = locate_entities(mesh, 2, boundary)
locate_fixed_BC = [locate_dofs_topological(W, 2, facets),]
bcs = [dirichletbc(u0,locate_fixed_BC)]

########## Solve with Newton solver wrapper: ##########
solveNonlinear(F,w,bcs,max_it=3,log=True)

########## Output: ##############

u_mid, _ = w.split()


print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(w.sub(0).sub(2).collapse().vector.getArray()))
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", len(w.vector.getArray()))
print("-"*50)

########## Visualization: ##############

u_mid, _ = w.split()


loc_str = voigt2D(gradv_local(w.sub(0), elastic_model.E01))
V0 = VectorFunctionSpace(mesh, ("CG", 1))
loc_str_func = Function(V0)
project(loc_str, loc_str_func)

with XDMFFile(MPI.COMM_WORLD, "solutions/shellmesh/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
with XDMFFile(MPI.COMM_WORLD, "solutions/shellmesh/loca_strain.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(loc_str_func)
