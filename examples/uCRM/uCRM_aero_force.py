"""
Structural analysis for the undeflected common research model (uCRM)
uCRM-9 Specifications: (units: m/ft, kg/lb)
(from https://deepblue.lib.umich.edu/bitstream/handle/2027.42/143039/6.2017-4456.pdf?sequence=1)
Maximum take-off weight	352,400kg/777,000lb
Wing span (extended)    71.75m/235.42ft
Overall length	        76.73m/251.75ft
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import locate_dofs_topological, Constant, dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import locate_entities
from dolfinx.io import VTXWriter

import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsX import *
from readCLT import readCLT, sortIndex


quad_mesh = ["uCRM-9_coarse.xdmf",
            "uCRM-9_medium.xdmf",
            "uCRM-9_fine.xdmf",]

test = 0
file_name = quad_mesh[test]
#file_name = tri_mesh[test]
path = "../../mesh/mesh-examples/uCRM-9-ICEM/"
mesh_file = path + file_name

with XDMFFile(MPI.COMM_WORLD, mesh_file, "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
# sometimes it should be `name="mesh"` to avoid the error
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

E = 7.31E10 # unit: (N/m^2)
nu = 0.3
h = 3E-3 # overall thickness (unit: m)
# rho = 2780 # density (kg/m^3)
# C = (E/(1.0 - nu*nu))*np.array([[1.0,  nu,   0.0         ],
#                                 [nu,   1.0,  0.0         ],
#                                 [0.0,  0.0,  0.5*(1.0-nu)]])
# A = h*C # Extensional stiffness matrix (3x3)
# D = h**3/12*C # Bending stiffness matrix (3x3)
# print("Homogenius stiffness matrix:")
# print("Membrane:", A)
# print("Bending:", D)
################### Thickness distribution ###################
hrcs = np.reshape(np.loadtxt(path+'uCRM_thickness_coarse.txt'),(nel,1))
h_ele = np.arange(nel)
cell_indices = mesh.topology.original_cell_index
h_array = sortIndex(hrcs, h_ele, cell_indices)
############# Apply element-wise thickness ###################
VT = FunctionSpace(mesh, ("DG", 0))
h = Function(VT)
h.vector.setArray(h_array)

########### Constant body force over the mesh ##########
# Scaled body force
# f_d = 2780*9.81 # force per unit area (unit: N/m^2)
# f_array = np.tile([0,0,f_d], nn)
# f = Constant(mesh,(0,0,f_d))*h # Body force per unit area

################### Aerodynamic loads ###################
################ Read and apply nodal force #################

frcs = np.reshape(np.loadtxt(path+'aero_force_coarse.txt'),(nn,3))
f_nodes = np.arange(nn)
# map input nodal indices to dolfinx index structure
coords = mesh.geometry.x
node_indices = mesh.geometry.input_global_indices
f_array = sortIndex(frcs, f_nodes, node_indices)

# print(frcs[0:10])
# print(f_array[0:10])
mesh.topology.create_connectivity(0, mesh.topology.dim)
cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
cell_ind = 1998
vertex_ind = cell_to_vertex.links(cell_ind)
print("Vertices id in the cell:", vertex_ind)
original_vertex_ind = np.zeros(4)
for i in range(4):
    print("Coordinates of "+str(i), coords[vertex_ind[i],:])
    original_vertex_ind[i] = node_indices[vertex_ind[i]]+1
print("Original vertices id in the cell:", original_vertex_ind.astype("int32"))
print(cell_indices[cell_ind]+1)
print(hrcs[cell_indices[cell_ind]], h_array[cell_ind])
# apply array in function space
VF = VectorFunctionSpace(mesh, ("CG", 1))
f = Function(VF)
f.vector.setArray(f_array) # Body force per unit area
####################################################
"""
CQUADR     23330     109   27933   27954   27955   27934     109
"""
########### Read the ABD matrices ##################

A,B,D,A_s = readCLT(path+'uCRM_ABD_coarse.txt')
clt_ele = np.arange(np.shape(A)[0])
A_ = sortIndex(A, clt_ele, cell_indices)
B_ = sortIndex(B, clt_ele, cell_indices)
D_ = sortIndex(D, clt_ele, cell_indices)
A_s_ = sortIndex(A_s, clt_ele, cell_indices)
CLT_data = (A_, B_, D_, A_s_)
####################################################
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


material_model = MaterialModelComposite(mesh,CLT_data)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
A = material_model.CLT[0]
elastic_energy = elastic_model.elasticEnergy(dx_inplane=dx_inplane,
                                            dx_shear=dx_shear)
#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)


############ Set the BCs for the airplane model ###################

u0 = Function(W)
u0.vector.set(0.0)

# Define BCs geometrically
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.less(x[1], 3.1))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.less(x[1], 3.1))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]


########## Solve with Newton solver wrapper: ##########
# solveNonlinear(F,w,bcs)

########## Output: ##############

uZ = computeNodalDisp(w.sub(0))[2]
print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(uZ))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))
print("  Number of total dofs = ", len(w.vector.getArray()))
print("-"*50)

########## Visualization: ##############
u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    u_mid.name = 'u_mid'
    xdmf.write_function(u_mid)
with XDMFFile(MPI.COMM_WORLD, "solutions/aero_loads.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    f.name = 'f'
    xdmf.write_function(f)
with XDMFFile(MPI.COMM_WORLD, "solutions/thickness.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    h.name = 'thickness'
    xdmf.write_function(h)
with XDMFFile(MPI.COMM_WORLD, "solutions/CLT_A.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    A.name = 'A'
    xdmf.write_function(A)

# VTE1 = ufl.FiniteElement("DG", mesh.ufl_cell(), 1)
# VT1 = FunctionSpace(mesh, VTE1)
# h1 = dolfinx.fem.Function(VT1)
# h1.interpolate(h)
# h1.name = 'thickness'

# with VTXWriter(mesh.comm, "solutions/CLT_A.bp", [A]) as vtx:
#     A.name = 'A'
#     vtx.write(0.0)
# with VTXWriter(mesh.comm, "solutions/h.bp", [h1]) as vtx:
#     vtx.write(0.0)

# from pathlib import Path
# def test_save_vtkx_cell_point(tempdir):
#     """Test writing point-wise data"""
#     mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 5)
#     P = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)

#     V = FunctionSpace(mesh, P)
#     u = Function(V)
#     u.interpolate(lambda x: 0.5 * x[0])

#     P1 = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)

#     V1 = FunctionSpace(mesh, P1)
#     u1 = Function(V1)
#     u1.interpolate(u)

#     u1.name = "A"

#     filename = Path(tempdir, "v.bp")
#     f = VTXWriter(mesh.comm, filename, [u1])
#     f.write(0)
#     f.close()

# test_save_vtkx_cell_point("solutions")