function [] = plot_displacement_field(ue, ex, ey, title_str)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
mag = 100; 
exd = ex + mag*ue(:,1:2:end); 
eyd = ey + mag*ue(:,2:2:end);

figure() 
patch(ex',ey',[0 0 0],'EdgeColor','none','FaceAlpha', 0.3) 
hold on 
patch(exd',eyd',[0 0 0],'FaceAlpha',0.3) 
axis equal 
title(title_str)

hold on
patch(ex',-ey'+1,[0 0 0],'EdgeColor','none','FaceAlpha', 0.3) 
hold on 
patch(exd',-eyd'+1,[0 0 0],'FaceAlpha',0.3) 
xlabel('x-position [cm]')
ylabel('y-position [cm]') 
end

