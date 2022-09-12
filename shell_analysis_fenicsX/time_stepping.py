from dolfinx.fem import Function, Constant
from shell_analysis_fenicsX import *
class ImplicitMidpointIntegrator:

    """
    Class to encapsulate backward Euler formulas for first- and second-order
    ODE systems.  
    """

    def __init__(self, mesh, DELTA_T, x, oldFunctions, t=0.0):
        """
        Initialize a backward Euler integrator with time step ``DELTA_T``.
        The unknown function is ``x``.  The sequence of ``Function``
        objects ``oldFunctions`` provides data from the previous time step.
        If ``oldFunctions`` contains only one ``Function``, then the system
        is assumed to be of order 1 and that function is interpreted as the
        initial value of ``x``.  If ``oldFunctions`` contains an additional
        element, then the ODE system is assumed to be of second order, and
        this additional element is interpreted as the initial velocity.
        The parameter ``t`` is the initial time, and defaults to zero.
        """
        self.systemOrder = len(oldFunctions)
        self.DELTA_T = DELTA_T
        self.x = x
        self.x_old = oldFunctions[0]
        self.mesh = mesh
        self.VX = self.x.function_space
        if(self.systemOrder == 2):
            self.xdot_old = oldFunctions[1]
        self.t = t + float(DELTA_T) # DELTA_T may be a Constant already
            
    def xdot(self):
        """
        Returns the approximation of the velocity at the current time step.
        """
        x_mid = 0.5*(self.x + self.x_old)
        return Constant(self.mesh, 2.0/self.DELTA_T)*self.x \
            - Constant(self.mesh, 2.0/self.DELTA_T)*self.x_old - self.xdot_old

    def xddot(self):
        """
        Returns the approximation of the acceleration at the current time
        step.
        """
        return Constant(self.mesh, 1.0/self.DELTA_T)*self.xdot() \
            - Constant(self.mesh, 1.0/self.DELTA_T)*self.xdot_old
            
    def advance(self):
        """
        Overwrites the data from the previous time step with the
        data from the current time step.
        """
        x_old = Function(self.VX)
        x_old.interpolate(self.x)
        if(self.systemOrder==2):
            xdot_old = Function(self.VX)
            project(self.xdot(), xdot_old)
        project(x_old, self.x_old)
        if(self.systemOrder==2):
            project(xdot_old, self.xdot_old)
        self.t += float(self.DELTA_T)

class BackwardEulerIntegrator:

    """
    Class to encapsulate backward Euler formulas for first- and second-order
    ODE systems.  
    """

    def __init__(self, mesh, DELTA_T, x, oldFunctions, t=0.0):
        """
        Initialize a backward Euler integrator with time step ``DELTA_T``.
        The unknown function is ``x``.  The sequence of ``Function``
        objects ``oldFunctions`` provides data from the previous time step.
        If ``oldFunctions`` contains only one ``Function``, then the system
        is assumed to be of order 1 and that function is interpreted as the
        initial value of ``x``.  If ``oldFunctions`` contains an additional
        element, then the ODE system is assumed to be of second order, and
        this additional element is interpreted as the initial velocity.
        The parameter ``t`` is the initial time, and defaults to zero.
        """
        self.systemOrder = len(oldFunctions)
        self.DELTA_T = DELTA_T
        self.x = x
        self.x_old = oldFunctions[0]
        self.mesh = mesh
        self.VX = self.x.function_space
        if(self.systemOrder == 2):
            self.xdot_old = oldFunctions[1]
        self.t = t + float(DELTA_T) # DELTA_T may be a Constant already
            
    def xdot(self):
        """
        Returns the approximation of the velocity at the current time step.
        """
        return Constant(self.mesh, 1.0/self.DELTA_T)*self.x \
            - Constant(self.mesh, 1.0/self.DELTA_T)*self.x_old

    def xddot(self):
        """
        Returns the approximation of the acceleration at the current time
        step.
        """
        return Constant(self.mesh, 1.0/self.DELTA_T)*self.xdot() \
            - Constant(self.mesh, 1.0/self.DELTA_T)*self.xdot_old
            
    def advance(self):
        """
        Overwrites the data from the previous time step with the
        data from the current time step.
        """
        x_old = Function(self.VX)
        x_old.interpolate(self.x)
        if(self.systemOrder==2):
            xdot_old = Function(self.VX)
            project(self.xdot(), xdot_old)
        project(x_old, self.x_old)
        if(self.systemOrder==2):
            project(xdot_old, self.xdot_old)
        self.t += float(self.DELTA_T)

