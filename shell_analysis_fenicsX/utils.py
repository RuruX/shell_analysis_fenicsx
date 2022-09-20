"""
The ``utils`` module
--------------------
Contains problem-specific functionalities such as project higher-order dofs
to a lower order function space (for plotting), and calculate the wing volume
of an airplane model.
"""

import dolfinx
import dolfinx.io
import ufl
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (set_bc, Function, FunctionSpace, form,
                        assemble_scalar, VectorFunctionSpace)
from ufl import TestFunction, TrialFunction, dx, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np



def project(v, target_func, bcs=[]):

    """
    Solution from
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-
    function-dolfinx/4142/6
    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = inner(Pv, w) * dx
    L = inner(v, w) * dx
    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)


def calculateSurfaceArea(mesh, boundary):

    #try to integrate a subset of the domain:
    Q = FunctionSpace(mesh, ("DG", 0))
    vq = TestFunction(Q)
    kappa = Function(Q)
    kappa.vector.setArray(np.ones(len(kappa.vector.getArray())))
    fixedCells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, boundary)

    with kappa.vector.localForm() as loc:
        loc.setValues(fixedCells, np.full(len(fixedCells), 0))

    s = assemble_scalar(form(vq*kappa*dx))
    surface_area = mesh.comm.allreduce(s, op=MPI.SUM)
    return surface_area

def computeNodalDisp(u):
    V = u.function_space
    mesh = V.mesh
    VCG1 = VectorFunctionSpace(mesh, ("CG", 1))
    u1 = Function(VCG1)
    u1.interpolate(u)
    uX = u1.sub(0).collapse().x.array
    uY = u1.sub(1).collapse().x.array
    uZ = u1.sub(2).collapse().x.array
    return uX,uY,uZ

def computeNodalDispMagnitude(u):
    uX, uY, uZ = computeNodalDisp(u)
    magnitude = np.zeros(len(uX))
    for i in range(len(uX)):
        magnitude[i] = np.sqrt(uX[i]**2+uY[i]**2+uZ[i]**2)
    return magnitude

class Delta:
    def __init__(self, x0, f_p, dist=1E-4, **kwargs):
        self.dist = dist
        self.x0 = x0
        self.f_p = f_p

    def eval(self, x):
        dist = self.dist
        values = np.zeros((3, x.shape[1]))
        for i in range(x.shape[1]):
            x_pt = np.array([x[0][i], x[1][i], x[2][i]])
            dist_ = np.linalg.norm(x_pt-self.x0)
            if dist_ < dist:
                values[0][i] = self.f_p[0]
                values[1][i] = self.f_p[1]
                values[2][i] = self.f_p[2]
        return values


def getCellID(coord, mesh):
    # get bbt for the mesh
    mesh_bbt = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    # convert point in array with one element
    points_list_array = np.array([coord, ])
    # for each point, compute a colliding cells and append to the lists
    points_on_proc = []
    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions(mesh_bbt, points_list_array)  # get candidates
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points_list_array)  # get actual
    for i, point in enumerate(points_list_array):
        if len(colliding_cells.links(i)) > 0:
            cc = colliding_cells.links(i)[0]
            points_on_proc.append(point)
            cells.append(cc)
    # convert to numpy array
    points_on_proc = np.array(points_on_proc)
    cells = np.array(cells)
    return cells
