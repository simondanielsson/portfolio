%% Stationary heat flow
clear all
clc

% Create mesh
load("petIMPORTANT.mat"); % pdetool % -> p e t

% Params
Tleft_day = 40;
Tleft_night = -96;

% Solve problem for day and night: calculate nodal temperatures
[a_day, K_day, f_day, ex, ey, eT_day, edof] = stat_heatflow(Tleft_day, p, e, t);
[a_night, K_night, f_night, ex, ey, eT_night, edof] = stat_heatflow(Tleft_night, p, e, t);

% Plot solution
plot_patch(ex, ey, eT_day, 'Stationary temperature distribution [C], day', hot); 
plot_patch(ex, ey, eT_night, 'Stationary temperature distribution [C], night', hot); 

% Calculate maximal temperatures
maxTempDay = max(a_day)
maxTempNight = max(a_night)

% Save variables
save("a_variables.mat", "a_day", "a_night", "K_day", "K_night", "f_day", "f_night", "ex", "ey", "eT_day", "eT_night", "edof");
%% b) Transient heat flow
clear all
clc

% Import data from a)
load("petIMPORTANT.mat");
load("a_variables.mat", "a_day", "a_night", "K_day", "K_night", "f_day", "f_night", "ex", "ey", "edof"); 

% Params
total_time = 4*60;
number_of_timesteps = 1000;
initial_value_day = a_night;
initial_value_night = a_day;

% Calculate nodal temperatures at all times
[aTot_day] = transient_heatflow(K_day, f_day, total_time, number_of_timesteps, initial_value_day, p, t);
[aTot_night] = transient_heatflow(K_night, f_night, total_time, number_of_timesteps, initial_value_night, p, t);

% Plot nodal temperatures at different times
number_of_figs = 2;
plot_transient_temps(aTot_day, edof, ex, ey, number_of_figs, total_time, number_of_timesteps, "day"); 
plot_transient_temps(aTot_night, edof, ex, ey, number_of_figs, total_time, number_of_timesteps, "night"); 
%% c) Mechanical problem: thermoelasticity (plane strain)
clear all
clc

% Import data from a)
load("petIMPORTANT.mat");
load("a_variables.mat", "eT_day", "eT_night", "ex", "ey"); 

[vm_day, u_day, ue_day, Edof] = mechanical_2d(eT_day, ex, ey, p, e, t);
[vm_night, u_night, ue_night, Edof] = mechanical_2d(eT_night, ex, ey, p, e, t);

% Plot effective stress
plot_patch(ex, ey, vm_day, "Effective von Mises stress due to ambient temperature, day", turbo)
plot_patch(ex, ey, vm_night, "Effective von Mises stress due to ambient temperature, night", turbo)

save("c_variables.mat", "u_day", "ue_day", "u_night", "ue_night", "Edof") 
%% d) Displacement fields
clear all
clc 

% Load needed data
load("petIMPORTANT.mat")
load("a_variables.mat", "ex", "ey");
load("c_variables.mat", "u_day", "u_night", "ue_day", "ue_night", "Edof");

[square_sum_day] = square_sum_displacements(u_day, Edof, p, t);
[square_sum_night] = square_sum_displacements(u_night, Edof, p, t);

% Plot displacement field
plot_displacement_field(ue_day, ex, ey, 'Displacement field [Magnitude enhancement 100], day');
plot_displacement_field(ue_night, ex, ey, 'Displacement field [Magnitude enhancement 100], night');
