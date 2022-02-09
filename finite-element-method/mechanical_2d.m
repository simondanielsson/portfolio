function [vonMises_nodal_perelement, u, ue, Edof] = mechanical_2d(eT, ex, ey, p, e, t)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Convert mesh into CALFEM quantities: e.g. edof 
nelm=length(t(1,:));
edof(:,1)=1:nelm;
edof(:,2:4)=t(1:3,:)';
coord=p';
ndof=max(max(t(1:3,:)));

% Params 
E_ti = 110e5; % N/cm2 
E_gl = 67e5;
nu_ti = 0.34;
nu_gl = 0.2;
expansion_ti = 9.4e-6;
expansion_gl= 7e-6;
numberOfNodes = length(coord);
numberOfDofs = 2*numberOfNodes;

% Construct new Edof
Edof = zeros(length(edof), 7);
for i=1:length(edof)
    Edof(i, 1) = i;
    Edof(i, 2) = 2*edof(i, 2) - 1;
    Edof(i, 3) = 2*edof(i, 2);
    Edof(i, 4) = 2*edof(i, 3) - 1;
    Edof(i, 5) = 2*edof(i, 3);
    Edof(i, 6) = 2*edof(i, 4) - 1;
    Edof(i, 7) = 2*edof(i, 4);
end

% Determine K-matrix and f0 (thermoelasticity force vector)

% Params
nu = 0;
E = 0;
T0 = 20; 
expansion = 0;
ep = [2 1];

% Initialize matrices/vectors
K = zeros(numberOfDofs);
f0 = zeros(numberOfDofs, 1);
epsilon_0 = zeros(numberOfDofs, 1);

% Iterate over all elements
for i=1:length(t)   
    % Determine constitutive matrix depending on region
    subdomain = t(4, i);
    if subdomain == 1
        E = E_ti;
        nu = nu_ti;
        expansion = expansion_ti;
    else 
        E = E_gl;
        nu = nu_gl;
        expansion = expansion_gl;
    end
    %D = E/(1+nu)/(1-2*nu)*[1-nu, nu, 0; nu, 1-nu, 0; 0, 0, 1/2*(1-2*nu)];
    D = hooke(2, E, nu); 
    
    % Element stiffness matrix
    Ke = plante(ex(i,:), ey(i,:), ep, D);
    
    % Element thermoelastic  
    epsilon_0 = expansion*(mean(eT(i,:)) - T0) * [1; 1; 1; 0];
    D_epsilon_0 = D*epsilon_0; % using average nodal temperature in element for deltaT. 
    f0e = plantf(ex(i,:), ey(i,:), ep, D_epsilon_0'); 
    
    % Assemble f0e:s, D_epsilon_0
    f0 = insert(Edof(i,:), f0, f0e);
    
    % Assemble element matrices
    K = assem(Edof(i,:), K, Ke);
end

% fb: hom neumann boundary gives zero integral. The other integral can also
% be set to zero, as either the corresponding dof has a zero shape function
% at the boundary (e.g. if the element does not share element boundary with
% global boundary) or the element traction vecotr is unknown, why we can
% set it to whatever as we know we have prescribed values (0) in a (vector). 

% Find dofs that lie on the homogenous Dirichlet (in x or y direction) boundary to create
% bc-vector 
dir_segments_y = [14 15 10 11 28 29 30]; %last 5 are from symmetry
dir_segments_x = [4 8 9];
edges_dir_x = []; 
edges_dir_y = []; 
er = e([1 2 5],:); % Reduced e

for i = 1:size(er,2)
    if ismember(er(3,i), dir_segments_x)
        edges_dir_x = [edges_dir_x er(1:2,i)]; % matrix with all nodes lying on a boundary with hom. dir. bcs. 
    elseif ismember(er(3,i), dir_segments_y)
        edges_dir_y = [edges_dir_y er(1:2,i)]; % matrix with all nodes lying on a boundary with hom. dir. bcs. 
    end
end

% Remove duplicate nodes 
unique_edges_dir_x = unique(edges_dir_x);
unique_edges_dir_y = unique(edges_dir_y);

dof_x = [];
dof_y = [];

% Convert the node labels into corresponding dofs
for i=1:length(unique_edges_dir_x)
    dof_x = [dof_x 2*unique_edges_dir_x(i)-1]; % rule to convert to dof for x: dof = 2*node-1
end
for i=1:length(unique_edges_dir_y)
    dof_y = [dof_y 2*unique_edges_dir_y(i)]; % rule to convert to dof for y: dof = 2*node
end

dirichlet_dofs = [dof_x dof_y]';

% Insert correct dofs into bc-vector

bc = [dirichlet_dofs zeros(length(dirichlet_dofs), 1)];

% Calculate nodal displacements
u = solveq(K, f0, bc);

% Get stresses and compute von Mises stress
ue = extract(Edof, u); % Element dof displacements
vonMises_element = zeros(length(t), 1);

for i=1:length(ue)
    % Check subdomain to determine constitutive matrix D
    subdomain = t(4, i);
    if subdomain == 1
        E = E_ti;
        nu = nu_ti; 
        expansion = expansion_gl;
    else 
        E = E_gl;
        nu = nu_gl;     
        expansion = expansion_gl;
    end
    D = hooke(2, E, nu); 
    epsilon_0 = expansion*(mean(eT(i,:)) - T0) * [1; 1; 1; 0];
    
    [es, et] = plants(ex(i,:), ey(i,:), ep, D, ue(i,:)); % es element stress
    es_temperature = es' - D*epsilon_0;
    vonMises_element(i) = sqrt(es_temperature(1)^2 + es_temperature(2)^2 + es_temperature(3)^2 - es_temperature(1)*es_temperature(2) - es_temperature(1)*es_temperature(3) - es_temperature(2)*es_temperature(3) + 3*es_temperature(4)^2); % + 3*es(5)^2 + 3*es(6)^2); % per element
end

% Convert to nodal vonMises
vonMises_nodal = zeros(numberOfNodes, 1); 

for i=1:size(coord,1) % for each node
    [c0, c1] = find(edof(:,2:4)==i);  % get elements that contains node i (in a vector)
    vonMises_nodal(i,1)=sum(vonMises_element(c0))/size(c0,1);  % calculate mean stress on node
end

% Put correct nodal von Mises stresses in vector, a row per element
vonMises_nodal_perelement = extract(edof, vonMises_nodal); 
end

