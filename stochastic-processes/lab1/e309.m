% Simulates a process with spectral density according to exercise 309.

% Anders Malmberg 20040207

f = 0:.001:5;
f = f(:);
R = zeros(size(f));
R(abs(f)<=1) = cos(pi*f(abs(f)<=1)/2).^2;

% use WAFO to simulate this process:
S = createspec([],'f');
S.S = R(f>=0);
S.f = f(f>=0);
subplot(311)
wspecplot(S)
X = spec2sdat(S,[],[],[],'random');
subplot(312)
plot(X(:,1),X(:,2))
grid
dt = 1/(2*max(f));
ti = sprintf('The process is sampled with d=%0.5g sec',dt);
title(ti)
% now we sample it too seldom
Y = X(1:1/dt:end,:);
subplot(313)
plot(Y(:,1),Y(:,2))
grid
title('The process is sampled with d = 1 sec');

xdata.x = X(:,2);
xdata.dt = dt;

ydata.x = Y(:,2);
ydata.dt = 1;

