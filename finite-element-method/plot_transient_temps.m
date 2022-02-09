function [] = plot_transient_temps(aTot, edof, ex, ey, number_of_figs, total_time, number_of_timesteps, day_or_night)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

plot_times_indeces = round(linspace(1, number_of_timesteps, number_of_figs));
times = linspace(0, total_time, number_of_timesteps);
for i=1:length(plot_times_indeces)
    eT = extract(edof, aTot(:,plot_times_indeces(i)));
    if day_or_night == "day"
        str_title = sprintf("Temperature distribution [C] after %0.1f seconds, day", times(plot_times_indeces(i)));
    else
        str_title = sprintf("Temperature distribution [C] after %0.1f seconds, night", times(plot_times_indeces(i)));
    end    
    plot_patch(ex, ey, eT, str_title, hot);
end
end

