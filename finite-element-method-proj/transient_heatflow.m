function [aTot] = transient_heatflow(K, f, total_time, number_of_timesteps, initial_value, p, t)
% PURPOSE: Calculates nodal temperatures of the body, between times 0 and total_time. 
%
% K: global stiffness matrix
% f: global force vector
% initial_value: field values at time 0

% 
% Convert mesh into CALFEM quantities: e.g. edof 
nelm=length(t(1,:));
edof(:,1)=1:nelm;
edof(:,2:4)=t(1:3,:)';
coord=p' ;
ndof=max(max(t(1:3,:)));

% Parameters
thickness = 1;
rho_c_gl = 3860e-6*670; % J/K cm3
rho_c_ti = 4620e-6*523;
numberOfNodes = length(coord);

% Compute C-matrix elementwise and assemble
C = zeros(numberOfNodes); 
rho_c = 0;
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
    
    posx = [x1 x2 x3];
    posy = [y1 y2 y3];
    
    % Get correct constant rho*c for corresponding subdomain
    if (subdomain == 1) % subdomain 1 is titanium domain
        rho_c = rho_c_ti;
    else 
        rho_c = rho_c_gl;
    end
    
    Ce = plantml(posx, posy, rho_c);
    
    indx = edof(i,2:end); 
    C(indx,indx) = C(indx,indx)+Ce;
end

% Time stepping parameters 
dt = total_time/number_of_timesteps;

% Implicit Euler, append to large vector aTot
aTot = zeros(numberOfNodes, number_of_timesteps); % [a1 a2 ... an], aj is a at time j
aTot(:,1) = initial_value;
for i=2:number_of_timesteps
    aTot(:, i) = (C+dt*K)\(C*aTot(:,i-1) + dt*f);
end

end

