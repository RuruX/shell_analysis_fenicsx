import numpy as np

node = []
element = []
x = []
y = []
z = []
node_count = 0
nl = 0
ele_count = 0
with open("crm_metallic_structure_grav_loads.bdf", "r") as f:
    lines = f.readlines()

    for l in lines:
        nl += 1
        if "CTRIA3" in l:
            ele_count += 1
            node_1 = int(l.split()[3])
            node_2 = int(l.split()[4])
            if l.split()[-1] == '+':
                
                next_l = lines[nl]
                if "$" in next_l:
                    break
                else:
                    node_3 = int(next_l.split()[1])
            else:
                node_3 = int(l.split()[5])
            element.append([node_1,node_2,node_3])
            
        if "GRID" in l:
            node_count += 1
            node.append(int(l.split()[1]))
            l_list = list(l)
            x_ = "".join((l_list[40:56]))
            y_ = "".join((l_list[56:72]))
            
            if '-' in y_[1:]:
                ind = y_[1:].find('-')
                y__ = float(y_[1:ind+1])/10**int(y_[ind+2:])
                y.append(y__)
            else:
                y.append(float(y_))
            x.append(float(x_))

            next_l = lines[nl]

            if "$" in next_l:
                break
            else:
                z.append(float(next_l.split()[1]))
            
coords = np.zeros((node_count,3))
coords[:,0] = np.array(x)
coords[:,1] = np.array(y)
coords[:,2] = np.array(z)
new_node = np.arange(node_count)

element = np.array(element)

print("Number of elements: ", str(ele_count))
print("Number of nodes: ", str(node_count))

node_used = []
for i in range(node_count):
    if i in element:
        node_used.append(i)
#    else:
#        print(i)
print("number of nodes used in the mesh:", len(node_used))
new_coords = coords[np.array(node_used)-1,:]

new_node = np.arange(len(node_used))
print(node_used[-1], new_node[-1])
# Re-ordering the node ID in the element dof table
for ele_i in range(element.shape[0]):
    for q_i in range(3):
        ind = np.where(node_used == element[ele_i,q_i])
        element[ele_i][q_i] = new_node[ind]

# Extra check of sanity:
#for i in range(len(node_used)):
#    if i not in element:
#        print("extra node:", i)
#        
#for ele_i in range(element.shape[0]):
#    for q_i in range(3):
#        if element[ele_i,q_i] not in new_node:
#            print("unknown node:", element[ele_i,q_i])

in2m = 0.0254 # 1 inch = 0.0254 meter

#np.savetxt('xyz_meter.txt', new_coords*in2m, fmt='%10.4f')

#np.savetxt('elements.txt', element, fmt='%8d')
