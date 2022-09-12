from csdl import Model
import csdl
from csdl.core.output import Output
import numpy as np


class CLT(Model):
    def initialize(self):
        # self.parameters.declare('h', types=float)
        self.parameters.declare('mat_prop')
        self.parameters.declare('no_plies')
        self.parameters.declare('ply_stack',default=np.array([]))
        self.parameters.declare('h0', default=0.0002) 

    def define(self):
        # h = self.parameters['h']      # you might want h to be constant for some applications
        mat_prop = self.parameters['mat_prop']
        no_plies = self.parameters['no_plies']
        ply_stack = self.parameters['ply_stack']
        h0 = self.parameters['h0']

        if ply_stack.size != no_plies:
            print(ply_stack.size)
            ply_stack = np.zeros(no_plies)

        h = self.declare_variable('h',shape=(1,),val=h0)   # can become array len(ply_stack) if desired
        ply_stack = self.create_input('ply_stack', shape=(no_plies,), val=ply_stack) # top to bottom
        self.add_design_variable('ply_stack', lower=-np.pi/2, upper=np.pi/2)

        q, q_out = self.calc_q(mat_prop)

        # I do this so that q is a variable type to work with matmat
        # I can't use create_output because the contents of q aren't
        # variables.
        q_var = self.declare_variable('q',val=q)
        q_out_var = self.declare_variable('q_out',val=q_out)

        q_bar = self.create_output('q_bar', shape = (3,3,no_plies))
        q_bar_out = self.create_output('q_bar_out', shape = (2,2,no_plies))
        for i in range(no_plies):
            t_eps = self.calc_t_eps(ply_stack[i],i)
            t_sig_inv = self.calc_t_sig_inv(ply_stack[i],i)
            aa = csdl.matmat(t_sig_inv,q_var)
            q_bar[:,:,i] = csdl.reshape(csdl.matmat(aa,t_eps),(3,3,1))
            
            c = csdl.cos(ply_stack[i])
            s = csdl.sin(ply_stack[i])
            q_bar_out[0,0,i] = csdl.reshape(q_out[0][0]*c**2 + q_out[1][1]*s**2,(1,1,1))
            q_bar_out[0,1,i] = csdl.reshape((q_out[1][1]-q_out[0][0])*s*c,(1,1,1))
            q_bar_out[1,0,i] = csdl.reshape((q_out[1][1]-q_out[0][0])*s*c,(1,1,1))
            q_bar_out[1,1,i] = csdl.reshape(q_out[0][0]*s**2 + q_out[1][1]*c**2,(1,1,1))
        A, B, D, A_star = self.calc_ABD(q_bar, q_bar_out, h, no_plies)
        A11 = -1*csdl.reshape(A[0,0],(1,))

        self.register_output('A11', A11)

        self.add_objective('A11')

    # Theis doesn't need to be in CSDL  
    def calc_q(self, mat_prop):
        E1 = mat_prop[0]
        E2 = mat_prop[1]
        v12 = mat_prop[2]
        G12 = mat_prop[3]
        v23 = mat_prop[4]

        G23 = E2/(2*(1+v23))
        D = 1-E2/E1*v12**2
        q = [[E1/D, v12*E2/D, 0], [v12*E2/D, E2/D, 0], [0, 0, G12]] # clt plane stress
        q_out = [[G23,0],[0,G12]] # out of plane (fsdt)
        # q = [[E1/D, v12*E2/D, 0, 0, 0], 
        #      [v12*E2/D, E2/D, 0, 0, 0], 
        #      [0, 0, G12, 0, 0],
        #      [0, 0, 0, G12, 0],
        #      [0, 0, 0, 0, G23]]
        return q, q_out

    # This and calc_t_eps are basically the same
    def calc_t_sig_inv(self, theta,i):
        C = csdl.cos(theta)
        S = csdl.sin(theta)
        Tsig_inv_array = [[C**2, S**2, -2*S*C],
                          [S**2, C**2, 2*S*C],
                          [S*C, -S*C, C**2-S**2]]

        t_sig_inv = self.create_output('t_sig_inv'+str(i),shape=(3,3))        # i makes sure the name is different each loop in define
        for i in range(3):                                                    # could replace with einsum -> faster?
            for j in range(3):
                t_sig_inv[i,j] = csdl.reshape(Tsig_inv_array[i][j],(1,1))
        # Tsig = [[C**2, S**2, 2*S*C], [S**2, C**2, -2*S*C], [-S*C, S*C, C**2-S**2]]
        return t_sig_inv

    def calc_t_eps(self, theta, i):
        C = csdl.cos(theta)
        S = csdl.sin(theta)
        TepsArray = [[C**2, S**2, S*C], [S**2, C**2, -S*C], [-2*S*C, 2*S*C, C**2-S**2]]
        t_eps = self.create_output('t_eps'+str(i),shape=(3,3))
        for i in range(3):
            for j in range(3):
                t_eps[i,j] = csdl.reshape(TepsArray[i][j],(1,1))    # again replace with einsum
        return t_eps

    # assembles the ABD matrices 
    def calc_ABD(self, q_bar, q_bar_out, h, no_plies):
        A = self.create_output('A',shape=(3,3))
        Al = np.zeros((3,3),dtype=Output)
        B = self.create_output('B',shape=(3,3))
        Bl = np.zeros((3,3),dtype=Output)
        D = self.create_output('D',shape=(3,3))
        Dl = np.zeros((3,3),dtype=Output)
        A_star = self.create_output('A_star',shape=(2,2))
        A_starl = np.zeros((2,2),dtype=Output)
        h = csdl.reshape(h,(1,1))
        for i in range(no_plies):
            z = h*(i+1-no_plies/2)
            a = h
            b = (z**2-(z-h)**2)/2
            d = (z**3-(z-h)**3)/3
            for j in range(3):
                for k in range(3):
                    Al[j,k] = Al[j,k] + csdl.reshape(q_bar[j,k,i],(1,1))*a
                    Bl[j,k] = Bl[j,k] + csdl.reshape(q_bar[j,k,i],(1,1))*b
                    Dl[j,k] = Dl[j,k] + csdl.reshape(q_bar[j,k,i],(1,1))*d
            for j in range(2):
                for k in range(2):
                    A_starl[j,k] = A_starl[j,k] + csdl.reshape(q_bar_out[j,k,i],(1,1))*a
        for i in range(3):
            for j in range(3):
                A[i,j] = Al[i,j]
                B[i,j] = Bl[i,j]
                D[i,j] = Dl[i,j]
        for i in range(2):
            for j in range(2):
                A_star[i,j] = A_starl[i,j]
        return A, B, D, A_star


#from csdl_om import Simulator
#import openmdao.api as om

## These are values from an old homework problem

##mat_prop = np.array([138*10**9, 10*10**9, .34, 7*10**9, 6*10**9])
#E = 3.0e6
#nu = 0.3
#G = E/2/(1+nu)
#mat_prop = np.array([E,E,nu,G,nu]) #isotropic material
#no_plies = 1
## ply_stack = np.radians(np.array([45, 0, -45, 90, 90, -45, 0, 45]))
## ply_stack = np.radians(np.array([21, 21, 21, 21, 21, 21, 21, 21]))
##ply_stack = 0.5*np.radians(np.array([90, 90, 90, 90, 90, 90, 90, 90]))
#ply_stack = 0.5*np.radians(np.array([90]))
#h = 3.0
##h = 0.0002



#sim = Simulator(CLT(no_plies=no_plies, mat_prop=mat_prop, ply_stack=ply_stack, h0=h))
## sim['ply_stack'] = ply_stack
#sim.run()
#print("A11", sim['A11'])
#print("ply_stack", sim['ply_stack'])
#print("A", sim['A'])
#print("B", sim['B'])
#print("D", sim['D'])
#print("A_star", sim['A_star'])


#G = E / 2 / (1 + nu)
#C = (E/(1.0 - nu*nu))*np.array([[1.0,  nu,   0.0         ],
#                             [nu,   1.0,  0.0         ],
#                             [0.0,  0.0,  0.5*(1.0-nu)]])
#A = h*C # Extensional stiffness matrix (3x3)
#B = 0 # Coupling (extensional bending) stiffness matrix (3x3)
#D = h**3/12*C # Bending stiffness matrix (3x3)
#A_s = G*h*np.eye(2) # # Out-of-plane shear stiffness matrix (2x2)
#print("A", A)
#print("B", B)
#print("D", D)
#print("A_star", A_s)
#prob = sim.prob

#prob.driver = om.ScipyOptimizeDriver()
#prob.driver.options['optimizer'] = 'SLSQP'

#prob.run_driver()

#print(sim['ply_stack'])
#print(sim['A11'])


