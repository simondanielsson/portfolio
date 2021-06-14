function [mag_squared] = square_sum_displacements(u, Edof, p, t)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

coord = p'; 

% -- Compute square of the sum of the displacements --
% Get elements and dofs in lefternmost lens
lens_elements = [];
lens_dofs = [];
for i = 1:length(t)
    subdomain = t(4,i);
    if(subdomain == 3)
        lens_elements = [lens_elements i];
        lens_dofs = [lens_dofs; Edof(i, 2:end)];
    end
end
unique_lens_dofs = unique(lens_dofs);

% All other dofs displacement values are zero, except those who lie in
% lens. 
u_lens = zeros(length(u),1);
for i = 1:length(unique_lens_dofs)
    u_lens(unique_lens_dofs(i)) = u(unique_lens_dofs(i));
end

% Get coords of nodes on every element, and compute Te elementwise. Then
% assemble
T = zeros(length(u));
for i=1:length(lens_elements)
    element = lens_elements(i);
   
    x1 = coord(t(1, element), 1);
    y1 = coord(t(1, element), 2);
    x2 = coord(t(2, element), 1);
    y2 = coord(t(2, element), 2);
    x3 = coord(t(3, element), 1);
    y3 = coord(t(3, element), 2);
    
    Te = plantml2d([x1 x2 x3], [y1 y2 y3], 1);
    
    % Assemble
    T = assem(Edof(element,:), T, Te);
end    

mag_squared = u_lens'*T*u_lens
end

