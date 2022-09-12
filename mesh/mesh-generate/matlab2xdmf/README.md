# Generate mesh files with Matlab code

This is the workflow to generate the user-defined mesh with tri/quad shell elements as the input for the problems using dolfin/dolfinx. One can start by changing the parameters in a Matlab script, and end up with getting a mesh file with format as ".xdmf"

The steps for generating quad mesh on a rectangular domain are as following:

1) choose the number of elements you want to use in x and y direction in the Matlab script `plate_quad_mesh.m` (**Nelx, Nely** in lines 8 and 9). 
2) open `mesh.xdmf`:
    - after `<DataItem DataType="Float" Dimensions="xxx 3" Format="XML" Precision="8">` copy the coordinates of the nodes (the **xyz_mat**) 
    - then after `<Topology NodesPerElement="4" NumberOfElements="xx" TopologyType="quadrilateral">
 <DataItem DataType="Int" Dimensions="xx 4" Format="XML" Precision="4">` copy paste the **M_touse** (it’s the connectivity matrix that positions the element node IDs counterclockwise).
    - and change the "Dimensions/NumberOfElements" correspondingly.
3) you can then call this mesh from your code as the input.
