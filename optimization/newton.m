function [X, N, optim] = newton(F, guess, tol, verbose) 
% Fp, Fpp is function first and second derivative of some C2 function F we want to minimize.   
% guess is a starting point for the algorithm. % Note: guessing at a stationary point of F will make the algorithm fail.
% tol = difference in subsequent function values in the sequence {f(x_k)}_k for stopping the script

% X output matrix containing all points and corresponding function values.  
% N = number of iterations
% optim = optimal point, i.e. in final row of X. 

% Method 

% Taylor expand f to second order and find the minimum point of the expansion. Take this point as
% the next point in the stepping algorithm. The point is given by x_k+1 = x_k - F'(x_k)/F''(x_k).   

% Solutions to non-positive second derivatives: 
% - Recompute derivatives at the new point
% - Recompute the derivatives with larger h < h_max
% and solution to method still yielding a larger function value at 
% the subsequent point, although second derivative is positive:
% - Recompute derivatives with smaller h > h_min. 

%------------------------------------------------------------------------------------------------------------

% Set default values if needed
DEFAULT_TOL = 1e-3;
if nargin < 3; tol = DEFAULT_TOL; end

DEFAULT_VERBOSE = 0;
if nargin < 4; verbose = DEFAULT_VERBOSE; end

if verbose
    disp("----------------------------------------------------")
    disp("----------------------------------------------------")
    disp("Executing line search...")
end 

% Compute derivatives 
[Fp, Fpp, is_stationary] = computeDerivatives(F, guess, verbose);

% Check if we guessed a stationary point, if function is constant, or if
% the second derivative is negative: then we cannot trust the method to yield a 
% optimal point
if is_stationary 
    optim = guess; 
    X = [guess, F(guess)]; 
    N = 1; 
    return 
end

% Perform line search
if verbose
    disp("Performing line search...");
end

[X, N, optim] = lineSearch(F, Fp, Fpp, guess, tol, verbose);

if isnan(F(optim))
    error('Bad job of the line search! F(optim) is NaN')
end

end


function [Fp, Fpp, is_stationary, h] = computeDerivatives(F, val, verbose)
% Computes the first and second (non-zero) derivative of function F at point guess. 

% Flag indicating if we guessed a stationary point
is_stationary = 0; 

% Compute derivatives
h = 1e-1;
Fp = differentiate(F, h);
Fpp = differentiate(Fp, h);

if verbose
    disp("Computing derivatives..."); 
    disp("First attempt: 1st derivative: " + Fp(val));
    disp("First attempt: 2nd derivative: " + Fpp(val));
    disp("--------------------------")
end

% Increment step size h if it yields a vanishing derivative 
h_max = 1e10; % stop if h becomes larger than this: then we probably cannot find a non-vanishing derivative 
tol_deriv = 1e-20; % tolerance for derivative being non-zero 
tol_sderiv = 1e-4; 
had_to_recompute = 0;

% Check if derivative is essentially zero or second derivative is essentially non-positive 
while abs(Fp(val)) < tol_deriv || Fpp(val) <= 0
    had_to_recompute = 1; 

    h = h*10; 
    Fp = differentiate(F, h);
    Fpp = differentiate(Fp, h); 

    if verbose
        disp("Increasing h to " + h); 
        disp("Now: 1st derivative: " + Fp(val));
        disp("Now: 2nd derivative: " + Fpp(val));
    end 

    if h > h_max
        % We cannot find a non-zero derivative or positive second derivative: 
        % the function is either constant, val is a stationary point or the function
        % is non-convex. Return corresponding flag.  
        if abs(Fp(val)) < tol_deriv 
            is_stationary = 1; 
            return;
        end
        if Fpp(val) <= 0 
            error("The function does not seem to be convex: cannot be minimized.");
        end
        return; 
    end
end

if verbose && had_to_recompute
    disp("--------------------------")
    disp("After iteration: 1st derivative =" + Fp(val));
    disp("After iteration: 2nd derivative = " + Fpp(val) + ",");
    disp("now using h = " + h);
    disp("--------------------------")
end

end


function [derivative] = differentiate(F, h)
% Calculates the derivative (function) of F to second order using step size h. 
    
derivative = @(x) (F(x+h) - F(x-h)) / (2*h);

end


function [X, N, optim] = lineSearch(F, Fp, Fpp, guess, tol, verbose)
% Performs a line search on F using Newtons method. 

% Initialization
X = [guess, F(guess)]; 
current_x = guess;
this_functionval = F(guess); % should not be zero 
previous_functionval = this_functionval/2; % TODO: choose such that first iteration is always performed

h_last = 1e-1; % Initial default differentiation step size: obviously not best idea to have to it hard coded here
h_min = 1e-20; % lower boundary for h

while abs(this_functionval - previous_functionval) > tol 

    if verbose
        disp("current_x = " + current_x);
        disp("F(current_x) = " + F(current_x));
        disp("Fp(current_x) = " + Fp(current_x));
        disp("Fpp(current_x) = " + Fpp(current_x));
        fprintf("\n");
    end

    if isinf(Fp(current_x)) || isinf(Fpp(current_x))
        [Fp, Fpp, h_new] = computeDerivativesSmallerH(F);

        if verbose
            disp("Derivatives inf: recomputing...")
            disp("Fp(current_x) = " + Fp(current_x));
            disp("Fpp(current_x) = " + Fpp(current_x));
            fprintf("\n");
        end
    end

    % If second derivative non-positive it will either crash, or
    % we won't get a lower function value at the next point.  
    % Recompute the derivatives
    while Fpp(current_x) <= 0; 
        if verbose
           disp("Second derivative <= 0: recomputing derivatives around new point..."); 
        end
        [Fp, Fpp, none, h_computed] = computeDerivatives(F, current_x, verbose);
        h_last = h_computed; 
    end

	% Compute next point in the sequence 
    next_x = current_x - Fp(current_x) / Fpp(current_x); 
    
    this_functionval = F(next_x); 
    previous_functionval = X(end, 2);      

    % Check if we still got a larger function value, even though the    
    % second derivative is positive. If so, recompute the derivatives with a smaller h and repeat
    if this_functionval > previous_functionval && h_last > h_min
        if verbose
            disp("func diff " + num2str(this_functionval - previous_functionval));
        end 

        [Fp_, Fpp_, h_new] = computeDerivativesSmallerH(F); 
        
        if verbose
            disp("decreasing h to " + h_new);
            disp("Fp(current_x) = " + Fp_(current_x));
            disp("Fpp(current_x) = " + Fpp_(current_x));
        end

        if Fpp_(current_x) <= 0
            if verbose 
                disp("this h " + num2str(h_new) + " yields a non-positive second derivative.");
                disp("Break and continue with the last computed derivatives");
            end 
            break; 
        end

        Fp = Fp_;
        Fpp = Fpp_;

        % Step
        next_x = current_x - Fp(current_x) / Fpp(current_x);
        this_functionval = F(next_x); 
        
        h_last = h_new; 

        if verbose
            disp("new x = " + next_x); 
            disp("new f diff = " + (this_functionval - previous_functionval)); 
            fprintf("\n");
        end
    end

    % Update quantities 
    X = [X; 
    	[next_x, this_functionval]];

    current_x = next_x; 
end 

if verbose
    disp("current_x = " + current_x);
    disp("F(current_x) = " + F(current_x));
    disp("Fp(current_x) = " + Fp(current_x));
    disp("Fpp(current_x) = " + Fpp(current_x));
    fprintf("\n");
end

% Fetch information about number of iterations N and the optimal point optim
N = size(X, 1); 
optim = X(end,1);
optimal_value = X(end, 2); 
stopping_cond_value = abs(this_functionval - previous_functionval); 

if verbose
    printDetails(optim, stopping_cond_value);
end 

end


function [Fp, Fpp, h_new] = computeDerivativesSmallerH(F)

h_new = 1e-6; % One step method

Fp = differentiate(F, h_new);
Fpp = differentiate(Fp, h_new);

end


function [] = printDetails(optim, stopping_cond_value) 

fprintf("Optimum found: %f\n", optim);
disp("--------------------------");
disp("Details:")
disp("Difference in function value from previous point:");
disp("delta f = " + stopping_cond_value);

end