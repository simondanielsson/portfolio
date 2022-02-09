function [x, k] = pen(method, x0, tol, restart)

% Objective function
f = @(x) exp(x(1)*x(2)*x(3)*x(4)*x(5));

% Conditions
h1 = @(x) x'*x - 10;
h2 = @(x) x(2)*x(3) - 5*x(4)*x(5);
h3 = @(x) x(1)^3 + x(3)^3 + 1; 

% Penalty parameters
p = 2; 
mu = @(k) 5*k; 

penalty = @(x) sum(power(abs([h1(x), h2(x), h3(x)]), p));

% Auxiliary function
f_aux = @(x, k) f(x) + mu(k)*penalty(x);

% Starting point
xprev = x0; 
xk = 2*x0; % Make some smart choice
k = 0; 

x = [x0]; 

printout = 0;

% Solve problem k
while norm(xk - xprev) > tol
    if k ~= 0
        xprev = xk; 
    end

    k = k + 1; 

    [xk, no_its, normg] = nonlinearmin(@(x) f_aux(x, k), xprev, method, 1e-4, restart, printout, 0);
    x = [x, xk];
end
