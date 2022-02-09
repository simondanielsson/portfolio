function [X, N, optim] = newtonTest(verbose) 

aTests(verbose); 
%easyTests(verbose); 

end


function [] = aTests(verbose)

f = @(lambda, a) (1 - 10^a*lambda)^2;
as = -[-10:10]; 

for const=as    
    g = @(lambda) f(lambda, const); 
    [X, N, optim] = newton(g, 0, 1e-3, verbose);

    printResultsA(f, optim, const, N);
end

end


function [] = easyTests(verbose)

% Test 1
funcs = {
    @(x) (x - 1)^2,
    @(x) exp(-x) + x^2, % Exercise 2.6
    @(x) exp(x^2) - 2*x % Exercise 2.7
};

test(funcs, verbose);

end


function [] = test(funcs, verbose)
% Perform test on functions f in funcs

for i=1:length(funcs)
    f = funcs{i};
    [X, N, optim] = newton(f, 0, 1e-3, verbose);
    printResults(f, optim); 
end

end


%% Auxiliary functions %% 

function [] = printResultsA(f, optim, a, N)

    disp("------------------------------------------------------");
    disp("Results:");
    disp("Function " + func2str(f) + ", a = " + a + ", has");
    fprintf("Optimal point: %d\n", optim);
    fprintf("Function value: %d\n", f(optim, a));
    fprintf("In %0.f iterations\n", N);
    fprintf("Line search finished successfully in %0.f iterations\n", N);
    
end

function [] = printResults(f, optim)

disp("------------------------------------------------------");
disp("Results:");
disp("Function " + func2str(f) + " has");
fprintf("Optimal point: %d\n", optim);
fprintf("Function value: %d\n", f(optim));
disp("Line search finished successfully.")
    
end




