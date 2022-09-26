%% 1
clear
clc 

load unknowndata

mean(data)

% The mean does not seem to be zero. We have to estimate the mean and
% subtract it from the data

x = data - mean(data);

mean(x)

X = fft(x);
n=length(x);
Rhat=(X.*conj(X))/n;
f=[0:n-1]/n;

plot(f-0.5,fftshift(Rhat))

% It looks like the signal has one component with approximately (discrete)
% frequency 0.2. However, we can't be sure that there isn't any other low
% power signal with a frequency around 0.2 that is being drowned by the
% leakage from the large peak. 


%% 2. 

N = 4096;
X=fft(x,N);
Rhat=(X.*conj(X))/n;
f=[0:N-1]/N;
plot(f,Rhat)

% It improvebs the frequency resolution of the estimate. 

% It should be a periodic function with approx frequency 0.2. 

%% 3

rhat=ifft(Rhat);
plot([0:15],rhat(1:16))

% As expected, the covariance function has frequency 0.2 (period time = 5).

%% 4.

e = randn(500,1);
modell.A=[1 -2.39 3.35 -2.34 0.96];
modell.C=[1 0 1];
x = filter(modell.C, modell.A, e);
plot(x)

[H,w]=freqz(modell.C,modell.A,2048);
R=abs(H).^2;
figure 
plot(w/2/pi,10*log10(R))

figure
periodogram(x,[],4096);

figure
periodogram(x,hanning(500),4096);

% When using the hanning window we can now observe the dip in spectral
% density around 0,25 Hz. Using the Fejér kernel (no data window) the dip
% is completely drowned by the leakage from the high power frequencies.
% This is due to the high amplitude sidelobes in the Fejer kernel. By using
% the hanning window, which has a lot quicker decreasing sidelobes, the
% leakage to nearby frequencies is a lot less (why the dip is not drowned
% by leakage). This is also called less biased. However, it also has a
% wider main lobe which means that leakage to adjacent frequencies is
% increased, which is seen as widened peaks everywhere. 

%% 

[Rhat,f]=periodogram(x,[],4096,1);
plot(f,Rhat) % Linear scale
figure
semilogy(f,Rhat) % Logarithmic scale

%%

K = 10;
n = length(x)
L = round(2*n/(K+1))

x = filter(modell.C, modell.A, e);
pwelch(x, hanning(L),[],4096);

% The variance is smaller (successive simulations yield similar results).
% We also see that we have very wide peaks, which corresponds to us using
% Hanning windows with wide mainlobes, thus increasing leakage to very
% local frequencies. 

%%
e = randn(500,1);
Rhate=periodogram(e,[],4096);
Rhatew=pwelch(e,hanning(L),[],4096);
var(Rhate)/var(Rhatew)

% the variance of the periodogram seems to be about twice as big as for the
% Welch method. 
% According to theory, the quotient should be equal to 1/(1/K)=K = 10 in
% our case, which it is not. 

%% 5 
clear 
clc

load eegdata12 

spekgui

% Data2 has a strong peak at 12 hz

%%
clear 
clc

load eegdatax

spekgui

% Data3 has a strong peak around 16 Hz.

%% 
clear
clc

load threeprocessdata
%%
plot(y1)
hold on
plot(y2)
hold on
plot(y3)
hold on

figure
plot(xcov(y1))
hold on
plot(xcov(y2))
hold on
plot(xcov(y3))
hold on

%% 

periodogram(y1,[],4096);
figure
periodogram(y2,[],4096);
figure
periodogram(y3,[],4096);

% Periodogram: seems to be a lot of leakage. It is difficult to see any
% distinct peaks which could make us identify the spectrum estimates with
% the real estimates. 

%% 
K = 10;
n = length(y1);
L = round(2*n/(K+1))

pwelch(y1, hanning(L),[],4096);
figure
pwelch(y2, hanning(L),[],4096);
figure
pwelch(y3, hanning(L),[],4096);

% 1a, 3b, 2c
% 2c, one sees the plateu. 
% In figure 1 we cannot see two peaks distinctly, however we can see one
% wide peak which is likely due to adjacent leakage from the two peaks,
% creating one wide peak (since hanning windows have a lot of very local
% leakage, wide peaks, due to wide main lobe).
% 3 has one distinct peak. 

%% 

mscohere(x1,y1,hanning(L),[],4096);
figure 
mscohere(x3,y1,hanning(L),[],4096);

% They seem to be mixes up, since the coherence spectrum is close to one
% for x3, y1. 
% The coherence spectrum provides a measure of the linear dependence
% between the amplitudes of x1 and y3. If it is one, they is directly
% proportional to each other. Since it is white noise, the probability of
% two independently generated white noise sequences being proportional for
% all frequencies is essentially zero, so it has to be mixed up. 
