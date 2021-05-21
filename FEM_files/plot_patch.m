function [] = plot_patch(ex, ey, eT, title_str, colormap_obj)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

figure() 
patch(ex',ey',eT') 
title(title_str) 
colormap(colormap_obj); 
colorbar; 
xlabel('x-position [cm]')
ylabel('y-position [cm]') 
axis equal

hold on 
patch(ex',-ey'+1,eT')
end

