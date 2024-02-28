"""
Modal analysis of a cantilever beam

-----------------------------------------------------------
Note: to run the example with the mesh files associated, you need to
have `git lfs` installed to download the actual mesh files. Please
refer to instructions on their official website at https://git-lfs.github.com/
-----------------------------------------------------------
"""
from logging import WARNING
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *


filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_2303_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local


# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
h = 0.003 # unit: m
rho = 2780 # unit: kg/m^3

f_0 = Constant(mesh, (0.0,0.0,0.0))

element_type = "CG2CG1"
element = ShellElement(
                mesh,
                element_type,
                )
W = element.W
w = Function(W)
u, theta = split(w)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear

#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh, w, material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)

dw = TestFunction(W)
du_mid,dtheta = split(dw)

dWint = elastic_model.weakFormResidual(elastic_energy, f_0)

# Inertial contribution to the residual:
dWmass = elastic_model.inertialResidual(rho, h)

######### Set the BCs to have all the dofs equal to 0 on the left edge ##########
# Define BCs geometrically
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.greater(x[1], -1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.greater(x[1], -1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]

K_form = derivative(dWint, w)
M_form = derivative(dWmass, w)
K = assemble_matrix(form(K_form), bcs)
M = assemble_matrix(form(M_form))
K.assemble()
M.assemble()

K_dense = K.convert("dense")
print(np.linalg.matrix_rank(K_dense.getDenseArray()))

from numpy.linalg import matrix_rank
M_dense = M.convert("dense")
print(np.linalg.matrix_rank(M_dense.getDenseArray()))
print(M_dense.getDenseArray().shape)

nev = 6

from slepc4py import SLEPc
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(K, M)
# 2 -- generalized Hermitian
eigensolver.setProblemType(2)
# nev -- number of eigenvalues
eigensolver.setDimensions(nev=nev)
# eigensolver.setWhichEigenpairs(4)
st = eigensolver.getST()
# ST --- spectral transformation object
# sinvert --- shift-and-invert
st.setType('sinvert')
st.setShift(0.)
st.setUp()
eigensolver.solve()
evs = eigensolver.getConverged()

eigenmodes = []
# Extraction

print( "Number of converged eigenpairs %d" % evs )
if evs > 0:
    for i in range(nev):
        # Extract eigenpair
        l = eigensolver.getEigenvalue(i)
        r = l.real
        # print(r)
        # vr -- the real part of the eigen vector
        vr, vi = K.createVecs()
        eigensolver.getEigenvector(i, vr, vi)

        # 3D eigenfrequency
        freq_3D = np.sqrt(r)/2/np.pi
        print("Shell FE: {:8.5f} [Hz]".format(freq_3D))

        # Initialize function and assign eigenvector
        eigenmode = Function(W,name="Eigenvector "+str(i))
        eigenmode.vector[:] = vr

        eigenmodes.append(eigenmode)

disp_modes = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./solutions_modal/disp_modes.xdmf", "w")
disp_modes.write_mesh(mesh)
rotation_modes = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./solutions_modal/rotation_modes.xdmf", "w")
rotation_modes.write_mesh(mesh)
for i in range(len(eigenmodes)):
    eigenvec = eigenmodes[i]
    disp, rotation = eigenvec.split()
    disp_modes.write_function(disp,i+1)
    rotation_modes.write_function(rotation,i+1)


#### Solid results #####
# Number of converged eigenpairs 9
# Solid FE:  8.18959 [Hz]   Beam theory:  8.07700 [Hz]
# Solid FE: 31.53203 [Hz]   Beam theory: 32.30801 [Hz]
# Solid FE: 50.82702 [Hz]   Beam theory: 50.61772 [Hz]
# Solid FE: 82.73509 [Hz]   Beam theory: 202.47086 [Hz]
# Solid FE: 140.17802 [Hz]   Beam theory: 141.73107 [Hz]
# Solid FE: 173.79316 [Hz]   Beam theory: 566.92428 [Hz]

#### Shell results ##### 40*200 mesh
# Number of converged eigenpairs 9
# Shell FE:  8.06830 [Hz]   E--B Beam theory:  8.07700 [Hz]
# Shell FE: 31.49654 [Hz]   E--B Beam theory: 32.30801 [Hz]
# Shell FE: 50.24086 [Hz]   E--B Beam theory: 50.61772 [Hz]
# Shell FE: 84.14625 [Hz]   E--B Beam theory: 202.47086 [Hz]
# Shell FE: 139.25773 [Hz]   E--B Beam theory: 141.73107 [Hz]
# Shell FE: 173.55986 [Hz]   E--B Beam theory: 566.92428 [Hz]