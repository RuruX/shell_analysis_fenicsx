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


beam = [#### quad mesh ####
        "plate_2_10_quad_1_5.xdmf",
        "plate_2_10_quad_2_10.xdmf",
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",
        "plate_2_10_quad_20_100.xdmf",
        "plate_2_10_quad_40_200.xdmf",
        "plate_2_10_quad_80_400.xdmf",]

filename = "../../mesh/mesh-examples/clamped-RM-plate/"+beam[5]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# E_val = 4.32e8
# nu_val = 0.0
E_val = 1e5 # unit: pa
nu_val = 0.
h_val = 0.5
width = 2.
length = 10.
rho_val = 1e-3

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
rho = Constant(mesh,rho_val)

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
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
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

# K_dense = K.convert("dense")
# print(K_dense.getDenseArray())
#
# from numpy.linalg import matrix_rank
# M_dense = M.convert("dense")
# print(matrix_rank(M_dense.getDenseArray()))
# print(M_dense.getDenseArray().shape)

from slepc4py import SLEPc
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(K, M)
# 2 -- generalized Hermitian
eigensolver.setProblemType(2)
# nev -- number of eigenvalues
eigensolver.setDimensions(nev=6)
# eigensolver.setWhichEigenpairs(4)
st = eigensolver.getST()
# ST --- spectral transformation object
# sinvert --- shift-and-invert
st.setType('sinvert')
st.setShift(0.)
st.setUp()
eigensolver.solve()
nev = 6
evs = eigensolver.getConverged()



# Exact solution computation
from scipy.optimize import root
from math import cos, cosh
from numpy import pi
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]
# # Set up file for exporting results
# xdmf = XDMFFile(MPI.COMM_WORLD, "solutions_modal/modal_analysis.xdmf", "w")
# xdmf.write_mesh(mesh)

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

        # Beam eigenfrequency
        if i % 2 == 0: # exact solution should correspond to weak axis bending
            I_bend = width*h_val**3/12.
        else:          #exact solution should correspond to strong axis bending
            I_bend = h_val*width**3/12.
        # print(I_bend)
        # print(rho_val*width*h_val*length)
        freq_beam = alpha(i/2)**2*np.sqrt(E_val*I_bend/(rho_val*width*h_val*length**4))/2/pi

        print("Shell FE: {:8.5f} [Hz]   E--B Beam theory: {:8.5f} [Hz]".format(freq_3D, freq_beam))

        # Initialize function and assign eigenvector
        eigenmode = Function(W,name="Eigenvector "+str(i))
        eigenmode.vector[:] = vr

        eigenmodes.append(eigenmode)
