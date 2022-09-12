import numpy as np

def sortIndex(frcs, f_nodes, dolfinx_indices):
    # map input nodal indices to dolfinx index structure
    rem_idcs = np.argsort(dolfinx_indices)
    f_nodes = rem_idcs[f_nodes]
    # initialize and populate full array
    row = len(dolfinx_indices)
    col = np.shape(frcs)[1]
    f_array = np.zeros((row,col))
    for ii in range(len(f_nodes)):
        f_array[f_nodes[ii],:] = frcs[ii,:]
    return f_array


def readCLT(filename):
    """
    Reads element-wise CLT matrices from specified file
    """

    A = []
    B = []
    D = []
    A_s = []

    nl = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for l in lines:
            A_i = []
            B_i = []
            D_i = []
            A_s_i = []
            nl += 1
            for i in range(9):
                A_i.append(float(l.split()[i]))
            for i in range(9,18):
                B_i.append(float(l.split()[i]))
            for i in range(18,27):
                D_i.append(float(l.split()[i]))
            for i in range(27,31):
                A_s_i.append(float(l.split()[i]))
            A.append(A_i)
            B.append(B_i)
            D.append(D_i)
            A_s.append(A_s_i)
    return (np.array(A).astype('float64'), np.array(B).astype('float64'),
            np.array(D).astype('float64'), np.array(A_s).astype('float64'))


# def readCLT(filename):
#     A = []
#     B = []
#     D = []
#     A_s = []

#     nl = 0
#     with open(filename, "r") as f:
#         lines = f.readlines()
#         # nl += 1
#         for l in lines:
#             nl += 1
#             for i in range(9):
#                 A.append(float(l.split()[i]))
#             for i in range(9,18):
#                 B.append(float(l.split()[i]))
#             for i in range(18,27):
#                 D.append(float(l.split()[i]))
#             for i in range(27,31):
#                 A_s.append(float(l.split()[i]))
#     return (np.array(A).astype('float64'), np.array(B).astype('float64'), 
#             np.array(D).astype('float64'), np.array(A_s).astype('float64'))