"""
The ``kinematics`` module
-----------------------
Contains basic matrix operations for basis transformations 
and voigt notation.
"""

from ufl import (grad, CellNormal, as_vector, Jacobian, sqrt, dot, cross, 
                as_matrix, indices, as_tensor)
     

def unit(v):

    """
    Normalize the vector ``v``.
    """
    
    return v/sqrt(dot(v,v))
    
def local_basis_inplane(mesh):

    """
    E2: Normal vector to each element is the third basis vector of the
        local orthonormal basis (indexed from zero for consistency with Python);
    E0: Local in-plane orthonormal basis vectors, with 0-th basis vector along
        0-th parametric coordinate direction (where Jacobian[i,j] is the partial
        derivatiave of the i-th physical coordinate w.r.t. to j-th parametric
        coordinate);
    """
    
    E2 = CellNormal(mesh)
    A0 = as_vector([Jacobian(mesh)[j,0] for j in range(0,3)])
    E0 = unit(A0)
    E1 = cross(E2,E0)

    return (E0,E1,E2)

def global_to_local_inplane(E0,E1):

    """
    Matrix for change-of-basis to/from local/global Cartesian coordinates,
    where E01[i,j] is the j-th component of the i-th basis vector:
    """
    
    T = as_matrix([[E0[i] for i in range(0,3)],
                   [E1[i] for i in range(0,3)]])

    return T                 

def gradv_local(v,T):

    """
    In-plane gradient components of displacement in the local orthogonal
    coordinate system:
    """
    
    gradv_global = grad(v) # (3x3 matrix, zero along E2 direction)
    i,j,k,l = indices(4)
    return as_tensor(T[i,k]*gradv_global[k,l]*T[j,l],(i,j))


def voigt2D(T,strain=True):

    """
    Convert a 2D symmetric rank-2 tensor ``T`` to Voigt notation.  If
    ``strain`` is true (the default), then the convention for strains is
    followed, where the off-diagonal component is doubled.
    """
    
    if(strain):
        fac = 2.0
    else:
        fac = 1.0
    return as_vector([T[0,0],T[1,1],fac*T[0,1]])

