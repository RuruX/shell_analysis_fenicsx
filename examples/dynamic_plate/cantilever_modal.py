#!/usr/bin/env python
# coding: utf-8

# # Modal analysis of an elastic structure
#
# This program performs a dynamic modal analysis of an elastic cantilever beam
# represented by a 3D solid continuum. The eigenmodes are computed using the
# **SLEPcEigensolver** and compared against an analytical solution of beam theory. We also discuss the computation of modal participation factors.
#
#
# The first four eigenmodes of this demo will look as follows:

# The first two fundamental modes are on top with bending along the weak axis (left) and along
# the strong axis (right), the next two modes are at the bottom.
#
# ## Implementation
#
# After importing the relevant modules, the geometry of a beam of length $L=20$
# and rectangular section of size $B\times H$ with $B=0.5, H=1$ is first defined:

# In[1]:

from logging import WARNING
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *


# L, B, H = 20., 0.5, 1.
# Nx = 200
# Ny = int(B/L*Nx)+1
# Nz = int(H/L*Nx)+1

L, B, H = 10., 2., 0.5

Nx = 200
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

pt1 = np.array([0.,0.,0.])
pt2 = np.array([L,B,H])
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [pt1, pt2], [Nx,Ny,Nz])


# Material parameters and elastic constitutive relations are classical (here we take $\nu=0$) and we also introduce the material density $\rho$ for later definition of the mass matrix:

# In[2]:

E_val = 1e5
nu_val = 0.
rho_val = 1e-3
E, nu = Constant(mesh,E_val), Constant(mesh,nu_val)
rho = Constant(mesh,rho_val)

# Lame coefficient for constitutive relation
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))
def sigma(v):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)


# Standard `FunctionSpace` is defined and boundary conditions correspond to a fully clamped support at $x=0$.

# In[3]:


V = dolfinx.fem.VectorFunctionSpace(mesh, ('CG', 1))
u_ = TrialFunction(V)
# u_ = Function(V)
du = TestFunction(V)

#
# def left(x, on_boundary):
#     return near(x[0],0.)
#
# bc = dirichletbc(V, Constant((0.,0.,0.)), left)

locate_BC = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.))
from petsc4py.PETSc import ScalarType
u_D = np.array([0,0,0], dtype=ScalarType)
bc = dirichletbc(u_D, locate_BC, V)
#
# # locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
# #                                     lambda x: np.less(x[1], 0.9858135))
# locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
#                                     lambda x: np.less(x[1], 1e-6))
# locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
#                                     lambda x: np.less(x[1], 1e-6))
# ubc=  Function(W)
# with ubc.vector.localForm() as uloc:
#      uloc.set(0.)
#
# bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
#         dirichletbc(ubc, locate_BC2, W.sub(1)),
#        ]

# The system stiffness matrix $[K]$ and mass matrix $[M]$ are respectively obtained from assembling the corresponding variational forms

# In[4]:


k_form = inner(sigma(du),eps(u_))*dx
l_form = Constant(mesh,1.)*u_[0]*dx
# K = PETScMatrix()
# b = PETScVector()
# assemble_system(k_form, l_form, bc, A_tensor=K, b_tensor=b)

m_form = rho*dot(du,u_)*dx
# M = PETScMatrix()
# assemble(m_form, tensor=M)
from femo.fea.utils_dolfinx import assembleSystem
# K, _ = assembleSystem(k_form, l_form, bcs=[bc])
K = assemble_matrix(form(k_form), [bc])
M = assemble_matrix(form(m_form))
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
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*np.pi/2.)['x'][0]

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
        # vr -- the real part of the eigen vector
        vr, vi = K.createVecs()
        eigensolver.getEigenvector(i, vr, vi)

        # 3D eigenfrequency
        freq_3D = np.sqrt(r)/2/np.pi

        if B <= H:
            # Beam eigenfrequency
            if i % 2 == 0: # exact solution should correspond to weak axis bending
                I_bend = H*B**3/12.
            else:          #exact solution should correspond to strong axis bending
                I_bend = B*H**3/12.
        else:
            # Beam eigenfrequency
            if i % 2 == 0: # exact solution should correspond to weak axis bending
                I_bend = B*H**3/12.
            else:          #exact solution should correspond to strong axis bending
                I_bend = H*B**3/12.
        freq_beam = alpha(i/2)**2*np.sqrt(E_val*I_bend/(rho_val*B*H*L**4))/2/np.pi

        print("Solid FE: {:8.5f} [Hz]   Beam theory: {:8.5f} [Hz]".format(freq_3D, freq_beam))

        # Initialize function and assign eigenvector
        eigenmode = Function(V,name="Eigenvector "+str(i))
        eigenmode.vector[:] = vr

        eigenmodes.append(eigenmode)


# The beam analytical solution is obtained using the eigenfrequencies of a clamped beam in bending given by $\omega_n = \alpha_n^2\sqrt{\dfrac{EI}{\rho S L^4}}$ where :math:`S=BH` is the beam section, :math:`I` the bending inertia and $\alpha_n$ is the solution of the following nonlinear equation:
#
# \begin{equation}
# \cos(\alpha)\cosh(\alpha)+1 = 0
# \end{equation}
#
# the solution of which can be well approximated by $(2n+1)\pi/2$ for $n\geq 3$.
#
# Since the beam possesses two bending axis, each solution to the previous equation is
# associated with two frequencies, one with bending along the weak axis ($I=I_{\text{weak}} = HB^3/12$)
# and the other along the strong axis ($I=I_{\text{strong}} = BH^3/12$). Since $I_{\text{strong}} = 4I_{\text{weak}}$ for the considered numerical values, the strong axis bending frequency will be twice that corresponding to bending along the weak axis. The solution $\alpha_n$ are computed using the
# `scipy.optimize.root` function with initial guess given by $(2n+1)\pi/2$.
#
# With `Nx=400`, we obtain the following comparison between the FE eigenfrequencies and the beam theory eigenfrequencies :
#
#Solid FE (shell):  2.43175 [Hz]   Beam theory:  2.01925 [Hz]
# Solid FE:  4.25887 [Hz]   Beam theory:  4.03850 [Hz]
# Solid FE: 15.19809 [Hz]   Beam theory: 12.65443 [Hz]
# Solid FE: 26.42822 [Hz]   Beam theory: 25.30886 [Hz]
# Solid FE: 42.37354 [Hz]   Beam theory: 35.43277 [Hz]
# Solid FE: 72.87733 [Hz]   Beam theory: 70.86554 [Hz]

# | Mode | Solid FE [Hz] | Beam theory [Hz] |
# | --- | ------ | ------- |
# | 1 |  2.04991 |  2.01925|
# | 2 |  4.04854 |  4.03850|
# | 3 | 12.81504 | 12.65443|
# | 4 | 25.12717 | 25.30886|
# | 5 | 35.74168 | 35.43277|
# | 6 | 66.94816 | 70.86554|
#

# ## Modal participation factors
#
# In this section we show how to compute modal participation factors for a lateral displacement in the $Y$ direction. Modal participation factors are defined as:
#
# \begin{equation}
# q_i = \{\xi_i\}[M]\{U\}
# \end{equation}
#
# where $\{\xi_i\}$ is the i-th eigenmode and $\{U\}$ is a vector of unit displacement in the considered direction. The corresponding effective mass is given by:
#
# \begin{equation}
# m_{\text{eff},i} = \left(\dfrac{\{\xi_i\}[M]\{U\}}{\{\xi_i\}[M]\{\xi_i\}}\right)^2 = \left(\dfrac{q_i}{m_i}\right)^2
# \end{equation}
#
# where $m_i$ is the modal mass which is in general equal to 1 for eigensolvers which adopt the mass matrix normalization convention.
#
# With `FEniCS`, the modal participation factor can be easily computed by taking the `action` of the mass form with both the mode and a unit displacement function. Let us now print the corresponding effective mass of the 6 first modes.

# In[8]:


# u = Function(V, name="Unit displacement")
# u.sub(1).collapse().x.array[:] = 1.
# combined_mass = 0
# from ufl import action
# for i, xi in enumerate(eigenmodes):
#     a = action(m_form, xi)
#     av = fem.assemble_vector(form(a))
#     qi = xi.vector*av
#     print(qi)
#     # qi = fem.assemble_scalar(form(action(action(m_form, u), xi)))
#     # exit()
#     mi = fem.assemble_scalar(form(action(action(m_form, xi), xi)))
#
#     meff_i = (qi / mi) ** 2
#     total_mass = fem.petsc.assemble_scalar(form(rho * dx(domain=mesh)))
#
#     print("-" * 50)
#     print("Mode {}:".format(i + 1))
#     print("  Modal participation factor: {:.2e}".format(qi))
#     print("  Modal mass: {:.4f}".format(mi))
#     print("  Effective mass: {:.2e}".format(meff_i))
#     print("  Relative contribution: {:.2f} %".format(100 * meff_i / total_mass))
#
#     combined_mass += meff_i
#
# print(
#     "\nTotal relative mass of the first {} modes: {:.2f} %".format(
#         N_eig, 100 * combined_mass / total_mass
#     )
# )
