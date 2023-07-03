"""
The ``shell_model`` module
--------------------------
Contains the most important classes of the R-M shell formulations:
the `materialModel` and the `elasticModel`, respectively for the
constitutive model and the elastic energy in weak form.
"""

from distutils.errors import DistutilsClassError

from numpy import float64
import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem import FunctionSpace, TensorFunctionSpace, Function, Constant
from dolfinx.nls.petsc import NewtonSolver
from ufl import (dx, inner, dot, cross, as_matrix, Identity, sym, split,
                CellDiameter, TestFunction, derivative, tr)
from shell_analysis_fenicsx.kinematics import *


class MaterialModel(object):

    """
    This class is the material model (also called the CLT model for composites)
    generator, which can either take the material properties to calculate
    the constitutive matrices (A|B|D|A_s for Reissner-Mindlin shell) for single-
    layer shell, or convert the user-defined CLT model from numpy arrays to ufl
    tensors for multi-layer materials.
    """

    def __init__(self, CLT=None, E=None, nu=None, h=None, BOT=False):
        self.CLT = CLT
        self.BOT = BOT
        if self.CLT is None:
            if (E and nu and h) is not None:
                self.CLT = self.getSingleLayerCLT(E,nu,h)
            else:
                raise ValueError("Material information is not complete.")
        else:
            self.CLT = self.convertToUFL(CLT)

    def convertToUFL(self, CLT):

        """
        Returns the constitutive matrices as ufl forms for composites
        """

        A,B,D,A_s = CLT
        return as_matrix(A), as_matrix(B), as_matrix(D), as_matrix(A_s)

    def getSingleLayerCLT(self,E,nu,h):

        """
        Returns the constitutive matrices for single-layer materials
        """

        G = E / 2 / (1 + nu)
        C = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                     [nu,   1.0,  0.0         ],
                                     [0.0,  0.0,  0.5*(1.0-nu)]])
        if self.BOT is True:
            A = h*C # Extensional stiffness matrix (3x3)
            B = -h**2/2*C # Coupling (extensional bending) stiffness matrix (3x3)
            D = h**3/3*C # Bending stiffness matrix (3x3)
            A_s = G*h*Identity(2) # # Out-of-plane shear stiffness matrix (2x2)
        else:
            A = h*C # Extensional stiffness matrix (3x3)
            B = 0 # Coupling (extensional bending) stiffness matrix (3x3)
            D = h**3/12*C # Bending stiffness matrix (3x3)
            A_s = G*h*Identity(2) # # Out-of-plane shear stiffness matrix (2x2)
        return (A,B,D,A_s)

class MaterialModelComposite(object):

    """
    This class is the material model (also called the CLT model for composites)
    generator, which can either take the material properties to calculate
    the constitutive matrices (A|B|D|A_s for Reissner-Mindlin shell) for single-
    layer shell, or convert the user-defined CLT model from numpy arrays to ufl
    tensors for multi-layer materials.
    """


    def __init__(self, mesh=None, CLT_data=None, E=None, nu=None, h=None):
        self.CLT_data = CLT_data
        self.mesh = mesh
        self.CLT = self.convertToUFL(CLT_data)

    def convertToUFL(self, CLT_data):
        """
        Returns the constitutive matrices as ufl forms for composites
        """
        VABD = TensorFunctionSpace(self.mesh, ("DG", 0), shape=(3, 3))
        VAs = TensorFunctionSpace(self.mesh, ("DG", 0), shape=(2, 2))
        A = Function(VABD)
        B = Function(VABD)
        D = Function(VABD)
        A_s = Function(VAs)

        A.vector.setArray(CLT_data[0])
        B.vector.setArray(CLT_data[1])
        D.vector.setArray(CLT_data[2])
        A_s.vector.setArray(CLT_data[3])
        return A, B, D, A_s

class ElasticModel(object):

    """
    Class for the Reissner-Mindlin shell model, which can generate the potential
    energy based on the given mesh, function space, and the material properties.
    """

    def __init__(self,mesh, w, clt_matrices, shl_offset=None):
        self.mesh = mesh
        self.w = w
        self.u_mid, self.theta = split(self.w)
        self.W = self.w.function_space
        E0,E1,self.E2 = local_basis_inplane(self.mesh)

        # Define thickness-based offsets depending on MID/BOT (or
        # any other) shell element reference plane definition
        off_fun = FunctionSpace(self.mesh, ("DG", 0))
        self.offset = Function(off_fun)
        if (shl_offset is not None):
            self.offset.vector.setArray(shl_offset)
        # otherwise defaults to zeros (MID)

        # Matrix for change-of-basis to/from local/global Cartesian coordinates;
        # E01[i,j] is the j-th component of the i-th basis vector:
        self.E01 = global_to_local_inplane(E0,E1)
        self.t_gu = gradv_local(self.u_mid,self.E01)
        self.A, self.B, self.D, self.A_s = clt_matrices
        self.isotropic = True
        if isinstance(self.A, dolfinx.fem.function.Function):
            self.isotropic = False
        self.kappa = self.local_bending_curvature()
        self.eps = self.local_membrane_strains()
        self.gamma = self.local_shear_strains()
        self.N, self.M, self.Q = self.computeStresses()

    def local_membrane_strains(self, offset=None):

        # user can pass in offsets directly in post-processing,
        # otherwise use total offset defined in elastic model
        if offset is None:
            offset = self.offset

        eps = sym(self.t_gu) - self.offset*self.kappa
        return eps

    def local_bending_curvature(self):
        kappa = sym(gradv_local(cross(self.E2, self.theta),self.E01))
        return kappa

    def local_shear_strains(self):

        """
        Transverse shear strains in local coordinates, as a vector
         such that gamma[i] = 2*eps[i,2], for i in {0,1}
        """

        dudxi2_global = -cross(self.E2,self.theta)
        i,j = indices(2)
        dudxi2_local = as_tensor(dudxi2_global[j]*self.E01[i,j],(i,))
        gradu2_local = as_tensor(
                        dot(self.E2,grad(self.u_mid))[j]*self.E01[i,j],(i,))
        gamma = dudxi2_local + gradu2_local
        return gamma

    def computeStresses(self):

        """
        Returns the stress tensors as the product of the CLT model and the
        local strains.
        """

        # membrane stresses
        N = self.A*voigt2D(self.eps) + self.B*voigt2D(self.kappa)
        # bending moments
        M = self.B*voigt2D(self.eps) + self.D*voigt2D(self.kappa)
        # out-of-plane shear stresses
        Q = self.A_s*self.gamma
        return N, M, Q

    def shearEnergy(self, dx_shear):
        return 0.5*dot(self.Q,self.gamma)*dx_shear

    def membraneEnergy(self, dx_inplane):
        return 0.5*dot(self.N,voigt2D(self.eps))*dx_inplane

    def bendingEnergy(self, dx_inplane):
        return 0.5*dot(self.M,voigt2D(self.kappa))*dx_inplane

    def drillingEnergy(self, E, h):
        h_mesh = CellDiameter(self.mesh)
        t_gu = self.t_gu

        drilling_strain = (self.t_gu[0, 1] - self.t_gu[1, 0]) / 2 + \
                                    dot(self.theta, self.E2)
        # these two scaling factors are consistent in unit
        if (not self.isotropic):
            alpha = max(self.D.vector.getArray())*12
        else:
            alpha = E*h**3
        drilling_stress = alpha*drilling_strain/h_mesh**2
        return 0.5*drilling_stress*drilling_strain*dx

    def elasticEnergy(self, E=None, h=None, dx_inplane=dx, dx_shear=dx):

        """
        Returns the potential energy of the elastic shell model.
        """

        return self.shearEnergy(dx_shear) + \
                self.membraneEnergy(dx_inplane) + \
                self.bendingEnergy(dx_inplane) + \
                self.drillingEnergy(E, h)

    def weakFormResidual(self, elasticEnergy, f,
                        penalty=False, g=None, dss=None, dSS=None):

        """
        Returns the PDE residual of the elasticity problem in weak form,
        where `f` is the applied body force per unit area.
        """

        dw = TestFunction(self.W)
        self.du_mid,self.dtheta = split(dw)
        retval = derivative(elasticEnergy,self.w,dw) - inner(f,self.du_mid)*dx
        if penalty:
            retval += self.penaltyResidual(self.w, dw, g, dss, dSS)
        return retval

    def penaltyResidual(self,u,v,g,dss,dSS):
        beta = Constant(self.mesh, 1E15)
        h_E = CellDiameter(self.mesh)
        n = CellNormal(self.mesh)
        return beta/h_E*inner(u-g, v)*dss + \
                (beta/h_E*inner(u-g, v))("+")*dSS + \
                (beta/h_E*inner(u-g, v))("-")*dSS

    def inertialResidual(self, rho, h):
        h_mesh = CellDiameter(self.mesh)
        retval = 0
        retval += rho*h*inner(self.u_mid, self.du_mid)*dx
        retval += rho*h*h_mesh**2*inner(self.theta, self.dtheta)*dx
        drilling_inertia = Constant(self.mesh,1.0)*h**3*dx
        # retval += drilling_inertia
        return retval

class ShellStressRM:
    """
    Class to compute Reissner-Mindlin shell's stresses using
    linear elastic material model.
    """
    def __init__(self, mesh, w, h_th, E, nu):
        """
        Parameters
        ----------
        w : dolfin Function. Numerical solution of problem.
        E : dolfin constant. Material's Young's modulus
        nu : dolfin constant. Material's Poisson's ratio
        h_th : dolfin constant. Shell's thickness
        """
        self.u_mid, self.theta = split(w)
        # Normal vector to each element is the third basis vector of the
        # local orthonormal basis (indexed from zero for consistency with Python):
        self.E2 = E2 = CellNormal(mesh)

        # Local in-plane orthogonal basis vectors, with 0-th basis vector along
        # 0-th parametric coordinate direction (where Jacobian[i,j] is the partial
        # derivatiave of the i-th physical coordinate w.r.t. to j-th parametric
        # coordinate):
        A0 = as_vector([Jacobian(mesh)[j,0] for j in range(0,3)])
        self.E0 = E0 = A0/sqrt(dot(A0,A0))
        self.E1 = E1 = cross(E2,E0)

        # Matrix for change-of-basis to/from local/global Cartesian coordinates;
        # E01[i,j] is the j-th component of the i-th basis vector:
        self.E01 = E01 = as_matrix([[E0[i] for i in range(0,3)],
                         [E1[i] for i in range(0,3)]])
        self.G = E / 2 / (1 + nu)
        self.D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                     [nu,   1.0,  0.0         ],
                                     [0.0,  0.0,  0.5*(1.0-nu)]])

    def u(self, xi2):
        """
        Displacement at through-thickness coordinate xi2:
        Formula (7.1) from http://www2.nsysu.edu.tw/csmlab/fem/dyna3d/theory.pdf
        """
        return self.u_mid - xi2*cross(self.E2,self.theta)


    def gradu_local(self, xi2):
        """
        In-plane gradient components of displacement in the local orthogonal
        coordinate system:
        """
        gradu_global = grad(self.u(xi2)) # (3x3 matrix, zero along E2 direction)
        i,j,k,l = indices(4)
        return as_tensor(self.E01[i,k]*gradu_global[k,l]*self.E01[j,l],(i,j))


    def eps(self, xi2):
        """
        In-plane strain components of local orthogonal coordinate system at
        through-thickness coordinate xi2, in Voigt notation:
        """
        eps_mat = sym(self.gradu_local(xi2))
        return as_vector([eps_mat[0,0], eps_mat[1,1], 2*eps_mat[0,1]])

    def gamma_2(self, xi2):
        """
        Transverse shear strains in local coordinates at given xi2, as a vector
        such that gamma_2(xi2)[i] = 2*eps[i,2], for i in {0,1}
        """
        dudxi2_global = -cross(self.E2,self.theta)
        i,j = indices(2)
        dudxi2_local = as_tensor(dudxi2_global[j]*self.E01[i,j],(i,))
        gradu2_local = as_tensor(dot(self.E2,grad(self.u(xi2)))[j]*self.E01[i,j],(i,))
        return dudxi2_local + gradu2_local

    def cauchyStresses(self, xi2):

        """
        Returns the constitutive matrices for isotropic materials
        """
        # out-of-plane shear
        sigma_shear = self.G*self.gamma_2(xi2)
        # in-plane stresses
        sigma_hat = self.D*self.eps(xi2)
        return (sigma_hat, sigma_shear)

    def vonMisesStress(self, xi2):
        """
        Returns Reissner-Mindlin shell's von Mises stress at the through
        thickness coordinate ``xi2`` (-h_th/2 <= xi2 <= h_th/2).
        """
        sigma_hat, sigma_shear = self.cauchyStresses(xi2)
        # von Mises stress formula with plane stress
        vonMises = sqrt(sigma_hat[0]**2 - sigma_hat[0]*sigma_hat[1]
                        + sigma_hat[1]**2 + 3*sigma_hat[2]**2
                        + 3*sigma_shear[0]**2 + 3*sigma_shear[1]**2)
        return vonMises


class DynamicElasticModel(object):

    """
    Class for the Reissner-Mindlin shell model, which can generate the potential
    energy based on the given mesh, function space, and the material properties.
    """

    def __init__(self,mesh,w,clt_matrices):
        self.mesh = mesh
        self.u_mid, self.theta = split(w)
        E0,E1,self.E2 = local_basis_inplane(self.mesh)

        # Matrix for change-of-basis to/from local/global Cartesian coordinates;
        # E01[i,j] is the j-th component of the i-th basis vector:
        self.E01 = global_to_local_inplane(E0,E1)
        self.t_gu = gradv_local(self.u_mid,self.E01)
        self.A, self.B, self.D, self.A_s = clt_matrices
        self.isotropic = True
        if isinstance(self.A, dolfinx.fem.function.Function):
            self.isotropic = False
        self.eps = self.local_membrane_strains()
        self.kappa = self.local_bending_curvature()
        self.gamma = self.local_shear_strains()
        self.N, self.M, self.Q = self.computeStresses()

    def local_membrane_strains(self):
        eps = sym(self.t_gu)
        return eps

    def local_bending_curvature(self):
        kappa = sym(gradv_local(cross(self.E2, self.theta),self.E01))
        return kappa

    def local_shear_strains(self):

        """
        Transverse shear strains in local coordinates, as a vector
         such that gamma[i] = 2*eps[i,2], for i in {0,1}
        """

        dudxi2_global = -cross(self.E2,self.theta)
        i,j = indices(2)
        dudxi2_local = as_tensor(dudxi2_global[j]*self.E01[i,j],(i,))
        gradu2_local = as_tensor(
                        dot(self.E2,grad(self.u_mid))[j]*self.E01[i,j],(i,))
        gamma = dudxi2_local + gradu2_local
        return gamma

    def computeStresses(self):

        """
        Returns the stress tensors as the product of the CLT model and the
        local strains.
        """

        # membrane stresses
        N = self.A*voigt2D(self.eps) + self.B*voigt2D(self.kappa)
        # bending moments
        M = self.B*voigt2D(self.eps) + self.D*voigt2D(self.kappa)
        # out-of-plane shear stresses
        Q = self.A_s*self.gamma
        return N, M, Q

    def plane_stress_elasticity(self, E, nu):
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)
        return lmbda_ps * tr(self.eps) * Identity(2) + 2 * mu * self.eps

    def shearEnergy(self, dx_shear):
        return 0.5*dot(self.Q,self.gamma)*dx_shear

    def membraneEnergy(self, dx_inplane):
        return 0.5*dot(self.N,voigt2D(self.eps))*dx_inplane

    def bendingEnergy(self, dx_inplane):
        return 0.5*dot(self.M,voigt2D(self.kappa))*dx_inplane

    def drillingEnergy(self, E, h):
        h_mesh = CellDiameter(self.mesh)

        drilling_strain = (self.t_gu[0, 1] - self.t_gu[1, 0]) / 2 + \
                                    dot(self.theta, self.E2)
        # these two scaling factors are consistent in unit
        if (not self.isotropic):
            alpha = max(self.D.vector.getArray())*12
        else:
            alpha = E*h**3
        drilling_stress = alpha*drilling_strain/h_mesh**2
        return 0.5*drilling_stress*drilling_strain*dx

    def elasticEnergy(self, E=None, h=None, dx_inplane=dx, dx_shear=dx):

        """
        Returns the potential energy of the elastic shell model.
        """

        return self.shearEnergy(dx_shear) + \
                self.membraneEnergy(dx_inplane) + \
                self.bendingEnergy(dx_inplane) + \
                self.drillingEnergy(E, h)

    def weakFormResidual(self, ALPHA, elasticEnergy, w, dw, f):

        """
        Returns the PDE residual of the elasticity problem in weak form,
        where `f` is the applied body force per unit area.
        """
        self.du_mid, self.dtheta = split(dw)
        return Constant(self.mesh,1/ALPHA)*derivative(elasticEnergy,w,dw) - inner(f,self.du_mid)*dx

    def inertialResidual(self, rho, h, uddot, thetaddot):
        retval = rho*(h*inner(uddot, self.du_mid)*dx
                    + h**3/12*dot(cross(self.E2, thetaddot),
                                cross(self.E2, self.dtheta))*dx)
        drilling_inertia = Constant(self.mesh,1.0)*h**3 \
                            *dot(dot(self.E2, thetaddot),
                                dot(self.E2, self.dtheta))*dx
        # retval += drilling_inertia
        return retval


def solveNonlinear(F, w, bcs, abs_tol=1e-50, max_it=3, log=False):

    """
    Wrap up the nonlinear solver for the problem F(w)=0 and
    returns the solution
    """

    problem = NonlinearProblem(F, w, bcs)

    # Set the initial guess of the solution
    with w.vector.localForm() as w_local:
        w_local.set(0.1)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    if log is True:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    # Set the Newton solver options
    solver.atol = abs_tol
    solver.max_it = max_it
    solver.error_on_nonconvergence = False
    opts = PETSc.Options()
    opts["nls_solve_pc_factor_mat_solver_type"] = "mumps"
    solver.solve(w)


def solveKSP(A, b, x):
    """
    Wrap up the KSP solver for the linear system Ax=b
    """
    ######### Set up the KSP solver ###############

    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)

    # additive Schwarz method
    pc = ksp.getPC()
    pc.setType("asm")

    ksp.setFromOptions()
    ksp.setUp()

    localKSP = pc.getASMSubKSP()[0]
    localKSP.setType(PETSc.KSP.Type.GMRES)
    localKSP.getPC().setType("lu")
    localKSP.setTolerances(1.0e-12)
    #ksp.setGMRESRestart(30)
    ksp.setConvergenceHistory()
    ksp.solve(b, x)
    history = ksp.getConvergenceHistory()


def solveKSP_mumps(A, b, x):
    """
    Implementation of KSP solution of the linear system Ax=b using MUMPS
    """

    # setup petsc for pre-only solve
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")

    # set LU w/ MUMPS
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType('mumps')

    # solve
    ksp.setUp()
    ksp.solve(b, x)
