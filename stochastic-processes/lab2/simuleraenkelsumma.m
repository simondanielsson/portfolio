% Simulates a Gaussian process of the form
%
% X(t) = A0 + sum Ak cos(2 pi fk t + phik)
%
% described at p. 115-116

f = [5 10];          % frequencies
sigma2_ = [2 2];     % variances
N = 500;             % no of simulations
dt = 1/(2*max(f))/3; % sample distance
t = 0:dt:20;         % time
fprintf('Frequencies:\n')
disp(num2str(f))
fprintf('Variances:\n')
disp(num2str(sigma2_))
fprintf('The maximum frequency in a sampled signal: %0.5g Hz\n',1/(2*dt))
     
fprintf('Sampling distance: %0.5g sek\n',dt)
[rayamp]=enkelsumma(f,sigma2_,N,t);
%
% To use spekgui
clear data
data.x = rayamp(:,5);
data.dt = dt;
