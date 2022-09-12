"""
The ``elements`` module
-----------------------
Defines the finite element spaces with user-defined element type and
 quadrature degrees
"""

from dolfinx import *
from dolfinx.fem import FunctionSpace
from ufl import VectorElement, MixedElement, FiniteElement, dx

class ShellElement():

    """
    Set up the function space and the quadrature degrees on a given mesh
    """
    
    def __init__(self, mesh, element_type=None, 
                                inplane_deg=None, 
                                shear_deg=None):
        self.mesh = mesh
        self.cell = mesh.ufl_cell()
        if element_type == None:
            self.element_type = "CG2CG1" # default
        else:
            self.element_type = element_type
            
        self.W = self.setUpFunctionSpace()
        self.dx_inplane, self.dx_shear = \
                        self.getQuadratureRule(inplane_deg, shear_deg)
        
    
    def setUpFunctionSpace(self):
    
        """
        Set up function space and the order of integration, with the first 
        vector element being mid-surface displacement, and the second vector 
        element being linearized rotation.
        """
        
        mesh = self.mesh
        cell = self.cell
        element_type = self.element_type
        W = None
            
        if(element_type == "CG2CG1"):
            # ------ CG2-CG1 ----------
            VE1 = VectorElement("Lagrange",cell,1)
            VE2 = VectorElement("Lagrange",cell,2)
            WE = MixedElement([VE2,VE1])
            W = FunctionSpace(mesh,WE)
            
        # CG2CR1 for triangle elements only
        elif(element_type == "CG2CR1"):
            # ------ CG2-CR1 ----------
            VE2 = VectorElement("Lagrange",cell,2)
            VE1 = VectorElement("CR",cell,1)
            WE = MixedElement([VE2,VE1])
            W = FunctionSpace(mesh,WE)
            
        # Alnord-Falk (enriched elements) not supported by DOLFINX
        else:
            print("Invalid element type.")
            
        return W
    
    def getQuadratureRule(self, inplane_deg, shear_deg):

        """
        Returns the cell integrals for in-plane and shear energy with given
        quadrature degrees.
        """
        
        if inplane_deg == None and shear_deg == None:
            dx_shear = dx
            dx_inplane = dx
                
        else: 
            dx_inplane = dx(metadata={"quadrature_degree":inplane_deg})
            dx_shear = dx(metadata={"quadrature_degree":shear_deg})
        
        return dx_inplane, dx_shear
        
