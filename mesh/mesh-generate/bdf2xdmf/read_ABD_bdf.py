import numpy as np

import numpy as np
ele_mat = []
mat2 = []
mat2_id = []
pshell = []
ele_count = 0
mat2_count = 0
pshell_count = 0
nl = 0

def convertToFloat(string):
    string_ = "".join((string))
    value = float(string_)
    return value

with open("uCRM-9_wingbox_coarse.bdf", "r") as f:
    lines = f.readlines()

    for l in lines:
        nl += 1
        if "CQUADR" in l:
            ele_count += 1
            ele_mat.append([int(l.split()[1]),
                            int(l.split()[2])])


        if "MAT2" in l:
            mat2_count += 1
            l_list = list(l)    
            mat2_i = np.zeros((6,))
            mat2_id.append(int(l.split()[1]))
            mat2_i[0] = convertToFloat(l_list[24:40])
            mat2_i[1] = convertToFloat(l_list[40:56])
            mat2_i[2] = convertToFloat(l_list[56:72])
            next_l = lines[nl]
            next_l_list = list(next_l)
            mat2_i[3] = convertToFloat(next_l_list[8:24])
            mat2_i[4] = convertToFloat(next_l_list[24:40])
            mat2_i[5] = convertToFloat(next_l_list[40:56])
            mat2.append(mat2_i)
            # print(nl,mat2_i)

        if "PSHELL" in l:
            pshell_count += 1
            pshell_i = []
            # 1: pshell_id, 
            # 2: mat2_id(A), 
            # 3: thickness,
            # 4: mat2_id(D),
            # 5: bending moment of inertia,
            # 6: mat2_id(A_s),
            # 7: mat2_id(B)
            pshell_i.append(int(l.split()[1]))
            pshell_i.append(int(l.split()[2]))
            pshell_i.append(float(l.split()[3]))
            pshell_i.append(int(l.split()[4].strip("*")))
            next_l = lines[nl]
            pshell_i.append(float(next_l.split()[1]))
            pshell_i.append(int(next_l.split()[2]))
            next_ll = lines[nl+1]
            pshell_i.append(int(next_ll.split()[1]))
            pshell.append(pshell_i)




CLT = np.zeros((ele_count,31))
thickness = np.zeros(ele_count,)
thickness_ele = np.zeros(ele_count,)
def sortSecond(val):
    # sorts the array in ascending according to
    # first element
    return val[0]
 

ele_mat.sort(key=sortSecond)
def vec2Mat_ABD(vec):
    mat = np.zeros((3,3))
    mat[0,0] = vec[0]
    mat[0,1] = vec[1]
    mat[0,2] = vec[2]
    mat[1,0] = mat[0,1]
    mat[1,1] = vec[3]
    mat[1,2] = vec[4]
    mat[2,0] = mat[0,2]
    mat[2,1] = mat[1,2]
    mat[2,2] = vec[5]
    return mat

def vec2Mat_As(vec):
    mat = np.zeros((2,2))
    mat[0,0] = vec[0]
    mat[0,1] = vec[1]
    mat[1,0] = mat[0,1]
    mat[1,1] = vec[3]
    return mat

for ele in range(ele_count):
    pshell_id = ele_mat[ele][1]
    pshell_ele = pshell[pshell_id-1]
    ind_A = pshell_ele[1]-1
    ind_B = pshell_ele[6]-1
    ind_D = pshell_ele[3]-1
    ind_As = pshell_ele[5]-1
    h = pshell_ele[2]
    I = pshell_ele[4]*h**3/12
    A = h*vec2Mat_ABD(mat2[ind_A])
    B = vec2Mat_ABD(mat2[ind_B])
    D = I*vec2Mat_ABD(mat2[ind_D])
    A_s = vec2Mat_As(mat2[ind_As])
    # print(A)
    # print(B)
    # print(D)
    # print(A_s)
    # print(np.concatenate((A,B,D,A_s), axis=None))
    thickness[ele] = h
    CLT[ele,:] = np.concatenate((A,B,D,A_s), axis=None)

print(thickness[4727])
print("Number of elements: ", str(ele_count))
np.savetxt('uCRM_thickness_coarse.txt', thickness, fmt='%8.4f')
np.savetxt('uCRM_ABD_coarse.txt', CLT, fmt='%8.4f')



