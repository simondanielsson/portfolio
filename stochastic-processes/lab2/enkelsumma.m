function [rayamp]=enkelsumma(f,sigma2_,N,t,plotid)
% ENKELSUMMA simulates N realizations 
% of a stationary stochastic process where
%
% X(t) = A0 + sum Ak cos(2 pi fk t + phik)
%
% A0 is Gaussian distributed with variance sigma2(0)
% Ak is Rayleigh distributed with parameter sigma2(k)
% 
% Input,
% f       1-ggr-K vector with frequencies in growing order
% sigma2_ 1-ggr-K vector with corresponding variances 
% N       no of realizations
% t       time
% 
% Uotput,
% rayamp  The Gaussian process X(t)
%
% Example:
% f = [5 10];
% sigma2_ = [2 2];
% N = 1000;
% dt = 1/(2*max(f)+1);
% t = 0:dt:20;
% [rayamp,normamp]=enkelsumma(f,sigma2_,N,t);
%
% To use spekgui
% data.x = rayamp(:,5);
% data.dt = dt;

% Anders Malmberg 20040305


if nargin < 5
     plotid=1;
end

if sigma2_(1)==0
sigma2 = sigma2_(1:end);
else
sigma2 = sigma2_(:);
end


sigma2 = repmat(sigma2,1,N); % K-by-N matrix
%varR = sigma2*(4-pi)/2
R = raylrnd(sqrt(sigma2)); % K-by-N matrix
if sigma2_(1) == 0
   A0 = normrnd(0,sqrt(sigma2_(1)),length(f),1);
   R = [A0 R];
end
%var(R)
phi = 2*pi*rand(length(f),N);

% time interval
t = t(:);
T = repmat(t,1,N);
% signal
X = zeros(size(T));
for i = 1:length(f)
     r = R(i,:);
     r = repmat(r,size(T,1),1);
     p = phi(i,:);
     p = repmat(p,size(T,1),1);
     X = X+r.*cos(2*pi*f(i)*T + p);
end

if plotid
figure(1)
clf
subplot(221)
% spektrum
if f(1) == 0
   stem(f,[sigma2_(1) sigma2_(2:end)./2],'filled')
else
   stem(f,[sigma2_./2],'filled');
end
title('Spectral density');
xlabel('f')
ylabel('R(f)')
axis([min(f)-.5 max(f)+.5 0 max(sigma2_)+1])
grid
subplot(222)
tau = 0:.01:(1/min(f)*4);
r = zeros(size(tau));
for i = 1:length(f)
r = r + sigma2_(i)*cos(2*pi*f(i)*tau);
end
plot(tau,r)
grid on
title('Covariance function')
     xlabel('\tau')
     ylabel('r(\tau)')
subplot(223)
itend = (1/min(f)*5);
it = t<itend;
plot(t(it),X(it,1),t(it),X(it,2)+5,'r',t(it),X(it,3)-5,'black')
axis([0 t(sum(it)) -15 15])
title('Example of realizations')
xlabel('t')
ylabel('X(t)')
legend('X_1(t)','X_2(t)+5','X_3(t)-5')
grid on
subplot(224)
it = 45;
t_0 = (t(2)-t(1))*45;
normplot(X(45,:))
tit = sprintf('Gaussian plot of X(t) for t=%0.5g, %d realizations',t_0,N);
title(tit)
end

% Output
rayamp = X;
