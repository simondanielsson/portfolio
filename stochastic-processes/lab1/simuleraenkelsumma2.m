% Simulates a Gaussian process of the form 
%
% X(t) = A0 + sum Ak cos(2 pi fk t + phik)
%
% described at p. 115-116.
% Aliasing so the largest frequency is 12 Hz.

f = [16];           % frequencies
sigma2_ = [6];      % variances
N = 100;            % no of simulations
dt = 1/(1.5*16) ;   % sample distance (too long distances)
t = 0:dt:20;        % time
dtok = 1/(10*16);
tok = 0:dtok:20;
fprintf('Frequencies \n')
disp(num2str(f))
fprintf('Variances \n')
disp(num2str(sigma2_))
fprintf('The maximum frequency in a sampled signal:  %0.5g Hz\n',1/(2*dt))
fprintf('Sampling distance: %0.5g sek\n',dt)

[rayamp2]=enkelsumma(f,sigma2_,N,t,0);
clear data2
data.x = rayamp2(:,5);
data.dt = dt;
[rayamp2]=enkelsumma(f,sigma2_,N,tok);
