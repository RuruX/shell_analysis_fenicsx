clear all;
close all;
clc;

Lx = 20;
Ly = 2;
Lz = 2;

N = 8;
Nelx = Lx*N;
Nely = Ly*N;
Nelz = Lz*N;
Nel = Nelx*Nely + Nelx*Nelz;

dx = Lx/Nelx;
dy = Ly/Nely;
dz = Lz/Nelz;

Nnodes = (Nely+Nelz+1)*(Nelx+1);
Nnodesx = Nelx+1;
Nnodesy = Nely+1;
Nnodesz = Nelz+1;

x_vec = zeros(Nnodes,1);
y_vec = zeros(Nnodes,1);
z_vec = zeros(Nnodes,1);
xyz_mat = zeros(Nnodes,3);

vertex_ID = zeros(Nnodes,1);

M_quad = zeros(Nel,4);
M_tri = zeros(2*Nel,3);

node_number = 1;

%for elements on x-y plane
for j = 1:Nely+1
    for i = 1:Nelx+1
        
    x_vec(node_number,1) = (i-1)*dx;
    y_vec(node_number,1) = (j-1)*dy;
    z_vec(node_number,1) = 0;
    
    xyz_mat(node_number,1) = (i-1)*dx;
    xyz_mat(node_number,2) = (j-1)*dy;
    xyz_mat(node_number,3) = 0;
    
    vertex_ID(node_number,1) = node_number;
    
    node_number = node_number+1;

    end
end

node_xz_start = node_number;

%for elements on x-z plane
for j = 1:Nelz
    for i = 1:Nelx+1
        
    x_vec(node_number,1) = (i-1)*dx;
    y_vec(node_number,1) = Ly/2;
    z_vec(node_number,1) = -j*dz;
    
    xyz_mat(node_number,1) = (i-1)*dx;
    xyz_mat(node_number,2) = Ly/2;
    xyz_mat(node_number,3) = -j*dz;
    
    vertex_ID(node_number,1) = node_number;
    
    node_number = node_number+1;
    
    end
end

el_number = 1;

%for elements on x-y plane
for jj = 1:Nely
    for ii = 1:Nelx
        node1 = (jj-1)*Nnodesx+ii;
        node2 = (jj-1)*Nnodesx+ii+1;
        node3 = (jj)*Nnodesx+ii+1;
        node4 = (jj)*Nnodesx+ii;

        M_quad(el_number,1)=node1;
        M_quad(el_number,2)=node2;
        M_quad(el_number,3)=node3;
        M_quad(el_number,4)=node4;

        M_tri(2*el_number-1,1)=node1;
        M_tri(2*el_number-1,2)=node2;
        M_tri(2*el_number-1,3)=node4;
        M_tri(2*el_number,1)=node2;
        M_tri(2*el_number,2)=node3;
        M_tri(2*el_number,3)=node4;
        
        el_number=el_number+1;
        
    end
end

node_mid = Nely/2*(Nelx+1)+1;

% for elements at the T-junction
for ii = 1:Nelx
    node1 = node_mid+ii-1;
    node2 = node_xz_start+ii-1;
    node3 = node_xz_start+ii;
    node4 = node_mid+ii;
    
    M_quad(el_number,1)=node1;
    M_quad(el_number,2)=node2;
    M_quad(el_number,3)=node3;
    M_quad(el_number,4)=node4;

    M_tri(2*el_number-1,1)=node1;
    M_tri(2*el_number-1,2)=node2;
    M_tri(2*el_number-1,3)=node4;
    M_tri(2*el_number,1)=node2;
    M_tri(2*el_number,2)=node3;
    M_tri(2*el_number,3)=node4;
    
    el_number=el_number+1;
end


%for elements on x-z plane
for jj = 2:Nelz
    for ii = 1:Nelx

        node1 = (jj-2)*Nnodesx+node_xz_start+ii-1;
        node2 = (jj-1)*Nnodesx+node_xz_start+ii-1;
        node3 = (jj-1)*Nnodesx+node_xz_start+ii;
        node4 = (jj-2)*Nnodesx+node_xz_start+ii;

        M_quad(el_number,1)=node1;
        M_quad(el_number,2)=node2;
        M_quad(el_number,3)=node3;
        M_quad(el_number,4)=node4;

        M_tri(2*el_number-1,1)=node1;
        M_tri(2*el_number-1,2)=node2;
        M_tri(2*el_number-1,3)=node4;
        M_tri(2*el_number,1)=node2;
        M_tri(2*el_number,2)=node3;
        M_tri(2*el_number,3)=node4;

        el_number=el_number+1;
        
    end
end



M_quad_touse = M_quad-1;
M_tri_touse = M_tri-1;

figure
scatter3(x_vec,y_vec,z_vec)
xlabel('x')
ylabel('y')
zlabel('z')
