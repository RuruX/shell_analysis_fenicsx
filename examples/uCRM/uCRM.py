"""
Structural analysis for the undeflected common research model (uCRM)
uCRM-9 Specifications: (units: m/ft, kg/lb)
(from https://deepblue.lib.umich.edu/bitstream/handle/2027.42/143039/6.2017-4456.pdf?sequence=1)
Maximum take-off weight	352,400kg/777,000lb
Wing span (extended)    71.75m/235.42ft
Overall length	        76.73m/251.75ft
-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar, locate_dofs_topological, Constant, dirichletbc
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *

# icemmesh = ["uCRM-9_wingbox_quad_coarse.xdmf",
#             "uCRM-9_wingbox_quad_medium.xdmf",
#             "uCRM-9_wingbox_quad_fine.xdmf",]
icemmesh = ["uCRM-9_coarse.xdmf", #Tip deflection: 0.17673840967897833
            "uCRM-9_medium.xdmf", #Tip deflection: 0.1778104969452804
            "uCRM-9_fine.xdmf"] #Tip deflection: 0.1780523716805102

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

PATH_ICEM = "../../mesh/mesh-examples/uCRM-9-ICEM/"
test = 0

PATH = PATH_ICEM
file_name = icemmesh[test]

mesh_file = PATH + file_name

with XDMFFile(MPI.COMM_WORLD, mesh_file, "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
       mesh = xdmf.read_mesh(name="Grid")

# sometimes it should be `name="mesh"` to avoid the error


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
material_model = MaterialModel(E=E,nu=nu,h=h,BOT=True)
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
solveNonlinear(F,w,bcs)

########## Output: ##############

u_mid, _ = w.split()

with XDMFFile(MPI.COMM_WORLD, "output_finest_tri.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)

strain_energy = assemble_scalar(form(elastic_energy))
print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(w.sub(0).sub(2).collapse().vector.getArray()))

print("Total strain energy:", strain_energy)
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", len(w.vector.getArray()))
print("-"*50)

########## Visualization: ##############

u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/icemmesh/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
