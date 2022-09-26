%% 3.1
clc
clear

load data1

plot(data1.x);
% Q1: Looks like the process has at least close to zero mean.

m = mean(data1.x)
% The estimation seems to be close to zero. 


% Q2: if zero is contained in the 95% confidence interval, we can say that
% the process has zero mean with a level of significance of 5%. Elementary
% statistics yield the following confidence interval boundaries 
lambda = 1.96; 
std = std(data1.x);
n = length(data1.x);
boundaries = [m - lambda*std/sqrt(n), m + lambda*std/sqrt(n)]

% Zero is not contained in the confidence interval: we can say that the process
% does not have zero mean with confidence 95%. 

%% 3.2 
clc
clear

load covProc

tiledlayout(1, 3);

for k = 1:3
    nexttile;
    plot(covProc(1:end-k),covProc(k+1:end),'.')
    title(k);
end

% In the scatter plots, each point p = (y_t, y_(t-k)) for k = 1, 2, 3,
% respectively. The plots are mostly used for visualizing the covariance
% (which is related to the level of linear relation between the two sets 
% of samples), as if the covariance is large for time differences tau = k, y_t and
% y_(t-k) will be large at the same time, and small at the same time (and
% the opposite for negative covariance). Thus, in those cases the scatter 
% plots will have a shape close to a straight line. If the covariance is
% small, there a specific value of y_t does not say anything about the
% value of y_(t-k), and the scatter plot will be more spread out in a disc.

%% 
clc
clear

load covProc

[ycov,lags]=xcov(covProc,20,'biased')

% ycov(i) contains the covariance between y_t and y_(t-lags(i)), i.e. the 
% covariances for all time lags. 

plot(lags, ycov)
grid on
% As we expected, the covariance for k = +-2 is positive (scatter plot 
% almost forms a straight line with positive slope), negative for k =
% +-1 (straight line with negative slope), and close to zero for k = 3
% (scatter plot shows no linear relationship). 

%% 3.3
clc 
clear

simuleraenkelsumma

spekgui

% The peaks in the spectral distribution does not have equal height. For
% every new realization the heights are new and the one peak which is the
% tallest changes. 

% Possible explanation: 
% Since the process is modeled as a linear combination of cosines with
% amplitudes and phases being observations of some random variables, the
% expected values of these random variables affect the power of each
% frequency in the frequency domain. This can be seen mathematically as the
% delta functions having a height equal to the expected power of the
% amplitude of the cosines in the process. Since every time we get a new
% realization, we also get new observations of our random amplitudes and
% phases, which gives the different frequency components in the process
% different strengths (amplitudes) which in turn affects the height of the
% peak in the frequency domain (in the spectral density). 


%% 4.1   
clc 
clear

load cello 
load trombone

spekgui

% Cello: 18 peaks (perhaps more), main frequencies 220, 440, 660, 880, 1100, 1330... continues on

% Trombone: 6 peaks, main frequencies (use log scale to see) 220, 440, 660, 880,
% 1100, 1330 (last)

% Both samples have peaks at 50 Hz, likely coming as a capacitive disturbance 
% from the mains electricity which the tape recorder is connected to. 

%% 4.2 
clc
clear

load cello 

n = 2;
cello2.x=cello.x(1:n:end);
cello2.dt=cello.dt*n;

spekgui

% The spectrum is identical up to about 1320 Hz, where peaks are now
% appearing twice as often. The range of the spectrum is also now only [0,
% 2000] Hz, in comparison to 4000 Hz as the previous upper boundary. The
% "extra" peaks present at higher frequency are probably aliased since we
% halfed the sampling speed.

%% lower sampling, trombone
clc
clear

load trombone

n = 4;
trombone2.x = trombone.x(1:n:end);
trombone2.dt = trombone.dt*n;

spekgui

% Trombone: we are starting to see aliased peaks at n = 4, i.e. using a 
% fourth of the original sampling speed. 

%% lowpass
clc
clear

load cello

cello2.x=decimate(cello.x, 2);
cello2.dt=cello.dt*2;

spekgui

% Applying a lowpass filter to the signal before resamplig at a lower rate
% does not yield any aliased peaks as the peaks that would have been
% aliased have been removed from the signal. 






