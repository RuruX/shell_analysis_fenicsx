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

# lbf2newton = 4.44822
# fz = lbf2newton * np.array([
# 178.5249412,
# 225.7910602,
# 255.3444864,
# 254.0378545,
# 264.3659094,
# 274.6239472,
# 281.8637954,
# 292.5067646,
# 318.2693761,
# 325.1311971,
# 0,
# 324.954771,
# 318.1305384,
# 292.5699649,
# 281.8552967,
# 274.6799369,
# 264.4083816,
# 254.1059857,
# 255.3734613,
# 225.8430446,
# 178.5818996])

# f = np.zeros((len(fz),3))
# f[:,2] = fz
# pts = np.array([[ 2.94811022e+00,  4.26720000e+00,  6.04990180e-01],
#        [ 2.97757859e+00,  3.84045955e+00,  6.04825502e-01],
#        [ 3.00704575e+00,  3.41373953e+00,  6.04644661e-01],
#        [ 3.04885750e+00,  2.98700842e+00,  6.04464035e-01],
#        [ 3.09431722e+00,  2.56028844e+00,  6.04283473e-01],
#        [ 3.13977694e+00,  2.13356845e+00,  6.04102911e-01],
#        [ 3.18523667e+00,  1.70684846e+00,  6.03922349e-01],
#        [ 3.23069639e+00,  1.28012847e+00,  6.03741787e-01],
#        [ 3.28160445e+00,  8.74003116e-01,  6.03561200e-01],
#        [ 3.36157619e+00,  4.26719858e-01,  6.03381338e-01],
#        [ 3.36177427e+00,  1.50947138e-13,  6.03197467e-01],
#        [ 3.36156802e+00, -4.26719800e-01,  6.03378820e-01],
#        [ 3.28159740e+00, -8.73977233e-01,  6.03558999e-01],
#        [ 3.23068979e+00, -1.28011567e+00,  6.03739749e-01],
#        [ 3.18523061e+00, -1.70683566e+00,  6.03920480e-01],
#        [ 3.13977143e+00, -2.13355565e+00,  6.04101210e-01],
#        [ 3.09431225e+00, -2.56027564e+00,  6.04281941e-01],
#        [ 3.04885308e+00, -2.98699562e+00,  6.04462671e-01],
#        [ 3.00704183e+00, -3.41373123e+00,  6.04643452e-01],
#        [ 2.97757502e+00, -3.84045126e+00,  6.04824402e-01],
#        [ 2.94810822e+00, -4.26717128e+00,  6.04995104e-01]])

# # Only save left wing loads
# np.save("fz.npy", fz[11:])
# np.save("pts.npy", pts[11:,:])
# np.save("f_c.npy",f[11:,:])


# print(fz[11:])
# print(pts[11:,:])
# print(f[11:,:])
# exit()

f_c = np.load("f_c.npy")
x0 = np.load("pts.npy")
x0[:,1] += 0.1
# filename = "./pav_wing/pav_wing_6rib_caddee_mesh_2374_quad.xdmf"
filename = "./pav_wing/pav_wing_6rib_caddee_mesh_4862_quad.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    shell_mesh = xdmf.read_mesh(name="Grid")
nel = shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local
nn = shell_mesh.topology.index_map(0).size_local


# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
# h = 0.003 # unit: m

in2m = 0.0254
h = 0.05*in2m
rho = 2780 # unit: kg/m^3

f_0 = Constant(shell_mesh, (0.0,0.0,0.0))

element_type = "CG2CG1"
element = ShellElement(
                shell_mesh,
                element_type,
                )
W = element.W
w = Function(W)
u, theta = split(w)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


# from FSI_coupling.VLM_sim_handling import *
# from FSI_coupling.shellmodule_utils import *
# from FSI_coupling.NodalMapping import *
# from FSI_coupling.NodalMapping import *
# from FSI_coupling.mesh_handling_utils import *
# from FSI_coupling.array_handling_utils import *
# from FSI_coupling.shellmodule_csdl_interface import (
#                                 DisplacementMappingImplicitModel,
#                                 ForceMappingModel,
#                                 VLMForceIOModel,
#                                 VLMMeshUpdateModel
#                                 )


# # Define force functions and aero-elastic coupling object ########
# # print("x0 shape:", np.shape(x0))
# coupling_obj = FEniCSx_concentrated_load_coupling(shell_mesh, x0,
#                     W, RBF_width_par=2., RBF_func=RadialBasisFunctions.Gaussian)
# # print("G mat shape:", np.shape(coupling_obj.G_mat.map))
# # f_c_reordered = f_c.reshape(17,3).T.flatten()
# f_c_reordered = f_c.T.flatten()
# f_array = coupling_obj.compute_dist_solid_force_from_point_load(f_c_reordered)

from shell_pde import NodalMap, ShellPDE
from scipy.sparse.linalg import spsolve

pde = ShellPDE(shell_mesh)
def fmap(mesh, oml):
    G_mat = NodalMap(mesh, oml, RBF_width_par=4.0,
                        column_scaling_vec=pde.bf_sup_sizes)
    rhs_mats = G_mat.map.T
    mat_f_sp = pde.compute_sparse_mass_matrix()
    weights = spsolve(mat_f_sp, rhs_mats)
    return weights

weights = fmap(shell_mesh.geometry.x, x0)
f_array = np.dot(weights,f_c)
f_array_flatten = f_array.flatten()

# print(f_array.shape)
# apply array in function space
VF = VectorFunctionSpace(shell_mesh, ("CG", 1))
f = Function(VF)

##################### Verify the force vector ###################
fx_sum = 0.
fy_sum = 0.
fz_sum = 0.
f_c_sum = np.sum(f_c,axis=0)
VF0 = FunctionSpace(shell_mesh, ("CG", 1))
fx_func = Function(VF0)
fy_func = Function(VF0)
fz_func = Function(VF0)
fx_func.vector.setArray(f_array[:,0])
fy_func.vector.setArray(f_array[:,1])
fz_func.vector.setArray(f_array[:,2])
fx_sum = assemble_scalar(form(fx_func*dx))
fy_sum = assemble_scalar(form(fy_func*dx))
fz_sum = assemble_scalar(form(fz_func*dx))
# for i in range(nn):
#     fx_sum += f_array[3*i]
#     fy_sum += f_array[3*i+1]
#     fz_sum += f_array[3*i+2]
print("-"*60)
print("                               Projected CG1  "+"     Original     ")
print("-"*60)
print("Sum of forces in x-direction:", fx_sum, f_c_sum[0])
print("Sum of forces in y-direction:", fy_sum, f_c_sum[1])
print("Sum of forces in z-direction:", fz_sum, f_c_sum[2])
print("-"*60)
#################################################################
# exit()

f.vector.setArray(f_array) # Nodal force in Newton

dims = shell_mesh.topology.dim
# h_mesh = dolfinx.cpp.mesh.h(shell_mesh, dims, range(nel))
h_mesh = CellDiameter(shell_mesh)
#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(shell_mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)
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


########### Apply the point load #############################
#


f1 = Function(W)
f1_0,_ = f1.split()

# delta_mpt = Delta_mpt(x0=x0, f_p=f_c)
# f1_0.interpolate(delta_mpt.eval)
f1_0.interpolate(f)

f1_x = computeNodalDisp(f1.sub(0))[0]
f1_y = computeNodalDisp(f1.sub(0))[1]
f1_z = computeNodalDisp(f1.sub(0))[2]
print("-"*60)
print("                               Projected CG2   "+"     Original     ")
print("-"*60)
print("Sum of forces in x-direction:", np.sum(f1_x), f_c_sum[0])
print("Sum of forces in y-direction:", np.sum(f1_y), f_c_sum[1])
print("Sum of forces in z-direction:", np.sum(f1_z), f_c_sum[2])
print("-"*60)

# # Assemble linear system
# a = derivative(F,w)
# L = -F
# A = assemble_matrix(form(a), bcs)
# A.assemble()
# b = assemble_vector(form(L))
# # b_array = b.getArray()
# # b.setArray(f1.vector.getArray())
# # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# dolfinx.fem.set_bc(b, bcs)



# ######### Solve the linear system with KSP solver ############
# solveKSP_mumps(A, b, w.vector)
solveNonlinear(F,w,bcs)

u_mid, _ = w.split()

dofs = len(w.vector.getArray())

uZ = computeNodalDisp(w.sub(0))[2]
strain_energy = assemble_scalar(form(elastic_energy))


shell_stress_RM = ShellStressRM(shell_mesh, w, h, E, nu)
von_Mises_top = shell_stress_RM.vonMisesStress(h/2)
V1 = FunctionSpace(shell_mesh, ('CG', 1))
von_Mises_top_func = Function(V1)
project(von_Mises_top, von_Mises_top_func, lump_mass=False)

metadata = {"quadrature_degree": 4}
dxx = ufl.Measure("dx", domain=shell_mesh, metadata=metadata)

def max_vm_stress_cg1(vm_stress,dx,rho=200,alpha=None,m=1e-6):
    """
    Compute the maximum von Mises stress via p-norm
    `rho` is the Constraint aggregation factor
    """
    pnorm = (m*vm_stress)**rho*dx

    if alpha == None:
        # alpha is an estimation of the surface area
        # alpha_form = Constant(shell_mesh,1.0)*dx
        h_mesh = ufl.CellDiameter(shell_mesh)
        alpha_form = h_mesh**2/2*dx
        alpha = assemble_scalar(form(alpha_form))
    pnorm_val = 1/alpha*assemble_scalar(form(pnorm))
    max_vm_stress = 1/m*(pnorm_val)**(1/rho)
    # max_vm_stress = 1/alpha*pnorm_val
    return max_vm_stress

def max_vm_stress_exp(vm_stress,dx,rho=200,alpha=None,m=1e-6):
    """
    Compute the UFL form of then maximum von Mises stress via p-norm
    `rho` is the Constraint aggregation factor
    """
    pnorm = (m*vm_stress)**rho*dx

    if alpha == None:
        # alpha is a parameter based on the surface area
        alpha_form = Constant(shell_mesh,1.0)*dx
        alpha = assemble_scalar(form(alpha_form))

    pnorm_val = 1/alpha*assemble_scalar(form(pnorm))
    max_vm_stress_form = 1/m*(pnorm_val)**(1/rho)
    # max_vm_stress = 1/alpha*pnorm_val
    return max_vm_stress_form

def max_vm_stress_pre(vm_stress,dx,rho=200,alpha=None,m=1e-6):
    """
    Compute the UFL form of then maximum von Mises stress via p-norm
    `rho` is the Constraint aggregation factor
    """
    pnorm = (m*vm_stress)**rho*dx

    if alpha == None:
        # alpha is a parameter based on the surface area
        alpha_form = Constant(shell_mesh,1.0)*dx
        alpha = assemble_scalar(form(alpha_form))

    max_vm_stress_pre = 1/alpha*pnorm
    return max_vm_stress_pre

def max_vm_stress(vm_stress,dx,rho=200,alpha=None,m=1e-6):
    max_pre = max_vm_stress_pre(vm_stress,dx,rho,alpha,m)
    max_vm_stress = 1/m*(assemble_scalar(form(max_pre)))**(1/rho)
    return max_vm_stress

def dmax_vmdw(w,vm_stress,dx,rho=200,alpha=None,m=1e-6):
    max_vm_form = max_vm_stress_exp(vm_stress,dx,rho=200,alpha=None,m=1e-6)
    return derivative(max_vm_form, w)


# alpha is a parameter based on the cell area
h_mesh = ufl.CellDiameter(shell_mesh)
V1 = FunctionSpace(shell_mesh, ('CG', 1))
h_mesh_func = Function(V1)
project(h_mesh, h_mesh_func, lump_mass=False)
alpha = np.average(h_mesh_func.vector.getArray())**2/2

print("-"*50)
print("-"*8, "PAV wing with concentrated loads", "-"*9)
print("-"*50)
print("Tip deflection:", max(uZ))
# print("Tip deflection:", uZ_tip)
print("Total strain energy:", strain_energy)
print("Exact maximum von Mises stress:", np.max(von_Mises_top_func.vector.getArray()))
# print("Derivative of maximum von Mises stress wrt displacements:", np.linalg.norm(assemble_vector(dmax_vmdw).getArray()))
rho_list = [50, 100, 200]
# rho_list = [200]
print("rho     ", "Maximum von von Mises stress")
for rho in rho_list:
    print(rho, max_vm_stress(von_Mises_top,dx=dxx,rho=rho,m=1e-6, alpha=alpha))
    print(rho, max_vm_stress_exp(von_Mises_top,dx=dxx,rho=rho,m=1e-6, alpha=alpha))
print("  Number of elements = "+str(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local))
print("  Number of vertices = "+str(shell_mesh.topology.index_map(0).size_local))
print("  Number of total dofs = ", dofs)
print("-"*50)

with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid_tri.xdmf", "w") as xdmf:
    xdmf.write_mesh(shell_mesh)
    xdmf.write_function(u_mid)
with XDMFFile(MPI.COMM_WORLD, "solutions/von_Mises_top.xdmf", "w") as xdmf:
    xdmf.write_mesh(shell_mesh)
    xdmf.write_function(von_Mises_top_func)

project(f1_0,f)
with XDMFFile(MPI.COMM_WORLD, "solutions/distributed_force.xdmf", "w") as xdmf:
    xdmf.write_mesh(shell_mesh)
    xdmf.write_function(f)
