import numpy as np

node = []
element = []
x = []
y = []
z = []
node_count = 0
nl = 0
ele_count = 0
with open("../../Fine/FEM/uCRM-9_wingbox_fine.bdf", "r") as f:
    lines = f.readlines()

    for l in lines:
        nl += 1
        if "CQUADR" in l:
            ele_count += 1
            element.append([int(l.split()[3]),
                            int(l.split()[4]),
                            int(l.split()[5]),
                            int(l.split()[6])])
        if "GRID" in l:
            node_count += 1
            node.append(int(l.split()[1]))
            x.append(float(l.split()[3]))
            y.append(float(l.split()[4].strip("*")))
            next_l = lines[nl]

            if "$" in next_l:
                break
            else:
                z.append(float(next_l.split()[2]))

coords = np.zeros((node_count,3))
coords[:,0] = np.array(x)
coords[:,1] = np.array(y)
coords[:,2] = np.array(z)

new_node = np.arange(node_count)

element = np.array(element)

print("Number of elements: ", str(ele_count))
print("Number of nodes: ", str(node_count))

# Re-ordering the node ID in the element dof table
for ele_i in range(element.shape[0]):
    for q_i in range(4):
        ind = np.where(node == element[ele_i,q_i])
        element[ele_i][q_i] = new_node[ind]

np.savetxt('xyz.txt', coords, fmt='%8.4f')

np.savetxt('elements.txt', element, fmt='%8d')
