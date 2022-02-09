function [x, no_its, normg] = nonlinearmin(f, x0, method, tol, restart, printout, verbose)
% f: objective function: R^n -> R
% x0: starting point 
% method: 'DFP' or 'BFGS'
% tol: tolerance 
% restart: 0 or 1. If 0, the method will carry out n iterations (
%		which will yield an exact answer for quadratic functions). 
% 		Else the n iterations will be repeated until stopping 
% 		criterion is satisfied.     
% printout: 0 or 1. If 1, results from each iteration will be printed in console. 

% Check inputs
if restart ~= 1 && restart ~= 0
	error("", "restart must have value either 0 or 1", restart); 
end

if printout ~= 1 && printout ~= 0
	error("printout must have value either 0 or 1", restart); 
end

DEFAULT_VERBOSE = 0; 
if nargin < 7; verbose = DEFAULT_VERBOSE; end

% Initialize values
n = length(x0); % Number of dimensions
iteration = 0; % iteration counter
no_its = 0; 
xk = x0;  
gradf = zeros(n, 1); 
relativeFunctionValueDifference = tol + 1; 

if printout
	fprintf("Executing minimization of function %s\n", func2str(f));
	fprintf("Starting at point [");
	fprintf("%g, ", x0(1:end-1));
	fprintf("%g],\n", x0(end)); 
	fprintf("Using %s method.\n\n", method);

	fprintf("%s   %s     %s       %s     %s     %s  %s   %s\n","outer it.","iteration","x","step size","f(x)","norm(grad)","ls iters","lambda");
	fprintf("   %s               %+.2e               %.2e\n", "init", x0(1), f(x0));
	for j=2:n
		fprintf("                      %+.2e\n", x0(j)); 
	end
	fprintf("\n");
end

while relativeFunctionValueDifference > tol 
	% Count iterations  
	iteration = iteration + 1;
	
	if iteration == 1
		yj = x0;
	else	
		yj = xk;
	end 
	
	Dj = eye(n);
	
	% Inner loop: 
	for j = 1:n
		no_its = no_its + 1;

		% Compute next search direction
		gradf = grad(f, yj); 
		dj = -Dj*gradf;
		
		% Perform line search
		guess = 0;
		[X, lsIterations, lambdaj] = newton(@(lambda) f(yj + lambda*dj), guess, 1e-3, verbose);

		% Move to next point
		yj_next = yj + lambdaj*dj;

		% Print result
		if printout
			if j == 1; outer_it = int2str(iteration); else outer_it = " "; end; 
			fprintf("    %s           %.0f     %+.2e   %.2e    %.2e    %.2e       %.0f     %.2e\n", outer_it, j, yj_next(1), abs(lambdaj)*norm(dj), f(yj_next), norm(gradf), lsIterations, lambdaj);
			for i=2:n
				fprintf("                      %+.2e\n", yj_next(i));
			end
			fprintf("\n");
		end

		% If we are on the final iteration or we did not move in our last step
		if j == n || isequal(yj, yj_next)
			xk_next = yj_next;
			fx = X(end, 2); % TODO: kanske ta bort
			break; 
		else
			% Update Dj
			Dj = modifiedHessian(f, yj, yj_next, Dj, method);
			yj = yj_next; 
		end   
	end

	% Compute quantity used in termination condition. 
	% Note that we cannot compute termination condition if f(xk) == 0. 
	% If we cannot compute the /relative/ difference: compute the absolute difference
	fxk = f(xk);
	if fxk == 0 
		fxk = 1; 
	end

	relativeFunctionValueDifference = norm(f(xk_next) - f(xk)) / norm(fxk);
	xk = xk_next; 

	if ~restart
		break; 
	end
end

% Return values 
x = xk; 
normg = norm(gradf); 
