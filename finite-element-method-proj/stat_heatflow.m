function [a, K, f, ex, ey, eT, edof] = stat_heatflow(Tleft, p, e, t)
%Solve problem a)

% Init params
coord = p';
numberOfNodes = length(coord);
k_gl = 0.8e-2; % W/cm K
k_ti = 17e-2;
thickness = 1;
load = 0;
alpha_c = 100e-4; % W/cm2 K

% Convert mesh into CALFEM quantities: e.g. edof 
nelm=length(t(1,:));
edof(:,1)=1:nelm;
edof(:,2:4)=t(1:3,:)';
coord=p' ;
ndof=max(max(t(1:3,:)));
[Ex,Ey]=coordxtr(edof,coord,(1:ndof)',3);
eldraw2(Ex,Ey,[1,4,1])  
title("Mesh of half the camera used in the FE analysis")

% a) Stationary heat flow: Compute and assemble element stiffness matrices and element load vector (=0)
% Parameters
Kt = zeros(numberOfNodes); 
fl = zeros(numberOfNodes, 1);

for i=1:length(t)
    % Get node coordinates
    node1 = t(1, i);
    node2 = t(2, i);
    node3 = t(3, i);
    
    subdomain = t(4, i);
    
    x1 = coord(node1, 1);
    y1 = coord(node1, 2);
    x2 = coord(node2, 1);
    y2 = coord(node2, 2);
    x3 = coord(node3, 1);
    y3 = coord(node3, 2);
    
    ex = [x1 x2 x3];
    ey = [y1 y2 y3];
    
    % Get constitutive matrix
    if (subdomain == 1) % subdomain 1 is titanium domain
        k = k_ti;
    else 
        k = k_gl;
    end
    
    D = k*eye(2);
    
    % Calculate element stiffnes matrix for triangular element
    ep = thickness; 
    eq = load;
    [Kte, fe] = flw2te(ex, ey, ep, D, eq);
    
    % Assemble element matrices
    el = i; % element number (== edof(1)) 
    indx = edof(el,2:end); 
    Kt(indx,indx) = Kt(indx,indx)+Kte;
    fl(indx) = fl(indx) + fe;  
end

% Handling the convection term
% Check which segments that should have convections 
% Parameters
Tright = 20;    

er = e([1 2 5],:); % Reduced e

conv_segments_left = [1 2 20 31]; % Choosen boundary segments (left side)
conv_segments_right = [8 9]; % Choosen boundary segments 

edges_conv_left = []; % Matrix with node pair numbers that lie along (the left) convection boundary
edges_conv_right = [];
for i = 1:size(er,2)
    if ismember(er(3,i),conv_segments_left)
        edges_conv_left = [edges_conv_left er(1:2,i)]; 
    elseif ismember(er(3,i), conv_segments_right)
        edges_conv_right = [edges_conv_right er(1:2,i)]; 
    end
end

fb = zeros(numberOfNodes, 1); 

% Iterate over all element boundaries along convection sub-boundaries
% Start with left convection boundary
for i=1:length(edges_conv_left) 
    node1 = edges_conv_left(1, i);
    node2 = edges_conv_left(2, i);
    
    x1 = coord(node1, 1)
    y1 = coord(node1, 2);
    x2 = coord(node2, 1)
    y2 = coord(node2, 2);
    Li = sqrt((x1-x2)^2 + (y1-y2)^2) % length of the i:th convection sub-boundary
     
    integral = alpha_c*1/2*Li*Tleft*thickness; % boundary vector integral
    fb(node1) = fb(node1) + integral;
    fb(node2) = fb(node2) + integral;
end
edges_conv_left
edges_conv_right
% Right hand side convection boundary
for i=1:length(edges_conv_right) 
    node1 = edges_conv_right(1, i);
    node2 = edges_conv_right(2, i);
    
    x1 = coord(node1, 1);
    y1 = coord(node1, 2);
    x2 = coord(node2, 1);
    y2 = coord(node2, 2);
    Li = sqrt((x1-x2)^2 + (y1-y2)^2); % length of the i:th convection sub-boundary
     
    integral = alpha_c*1/2*Li*Tright*thickness; % boundary vector integral
    fb(node1) = fb(node1) + integral;
    fb(node2) = fb(node2) + integral;
end

% Calculating the M-matrix (from the convection boundary term)
M = zeros(numberOfNodes); 

% Iterate over all element boundaries along convection sub-boundaries
% Start with left convection boundary
for i=1:length(edges_conv_left) 
    node1 = edges_conv_left(1, i);
    node2 = edges_conv_left(2, i);
    
    x1 = coord(node1, 1);
    y1 = coord(node1, 2);
    x2 = coord(node2, 1);
    y2 = coord(node2, 2);
    Li = sqrt((x1-x2)^2 + (y1-y2)^2); % length of the i:th convection sub-boundary
    
    integral = alpha_c*thickness*1/6*Li*[2, 1; 1, 2]; % Easy integral of element shape function
    
    % Assemble
    M(node1, node1) = M(node1, node1) + integral(1, 1);
    M(node1, node2) = M(node1, node2) + integral(1, 2);
    M(node2, node1) = M(node2, node1) + integral(2, 1);
    M(node2, node2) = M(node2, node2) + integral(2, 2);  
end

% Now right hand side convection boundary
for i=1:length(edges_conv_right) 
    node1 = edges_conv_right(1, i);
    node2 = edges_conv_right(2, i);
    
    x1 = coord(node1, 1);
    y1 = coord(node1, 2);
    x2 = coord(node2, 1);
    y2 = coord(node2, 2);
    Li = sqrt((x1-x2)^2 + (y1-y2)^2); % length of the i:th convection sub-boundary
    
    integral = alpha_c*thickness*1/6*Li*[2, 1; 1, 2]; % Easy integral of element shape function
    
    % Assemble
    M(node1, node1) = M(node1, node1) + integral(1, 1);
    M(node1, node2) = M(node1, node2) + integral(1, 2);
    M(node2, node1) = M(node2, node1) + integral(2, 1);
    M(node2, node2) = M(node2, node2) + integral(2, 2);  
end

% Formulate full problem
K = Kt + M;
f = fb + fl; 

a = solveq(K, f);

% Plot
% Extract element temperatures
eT = extract(edof, a); 
ex = zeros(length(eT), 3); % numberOfElements long, number of nodes per element wide
ey = zeros(length(eT), 3); 

for i=1:length(edof)
    x1 = coord(edof(i,2), 1);
    y1 = coord(edof(i,2), 2);
    x2 = coord(edof(i,3), 1);
    y2 = coord(edof(i,3), 2);
    x3 = coord(edof(i,4), 1);
    y3 = coord(edof(i,4), 2);
    
    ex(i,:) = [x1 x2 x3]; 
    ey(i,:) = [y1 y2 y3]; 
end

end

