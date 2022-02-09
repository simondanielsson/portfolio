function [] = pen_test(restart, only_k)

disp("-----------------------------------TEST 1-----------------------------------")
x01 = [-2; 2; 2; -1; -1];
k = test_x0(x01, restart, only_k)
% Result: DFP and BFGS do not converge to same point for all tolerances: tolerance of 1 for BFGS yiels other point of convergence. 
% Smaller tolerance requires a larger number of iterations of xk (different mu_k) 
% before converging. In general, DFP requires more iterations both in the actual multi-dim search
% as well as for penalty-iterations xk (more different, and larger, mu_k required for conv).   

fprintf("\n"); fprintf("\n")
disp("-----------------------------------TEST 2-----------------------------------")
x02 = [-2; 3; 5; -1; -0.5];
k = test_x0(x02, restart, only_k)
% Result: Same convergence point for BFGS, DFP

fprintf("\n"); fprintf("\n")
disp("-----------------------------------TEST 3-----------------------------------")
x03 = -x01;
k = test_x0(x03, restart, only_k)
% Result: Both methods converge to similar points. 
% BFGS requires few more mu_k iterations for it to converge than DFP for tol=1e-3

fprintf("\n"); fprintf("\n")
disp("-----------------------------------TEST 4-----------------------------------")
x04 = [10; -5; 8; -0.5; -2];
k = test_x0(x04, restart, only_k)
% Result: Same point of convergence as x01

fprintf("\n"); fprintf("\n")
disp("-----------------------------------TEST 5-----------------------------------")
x05 = -x04;
try 
    k = test_x0(x05, restart, only_k)
catch ME
    warning(ME.message);
end 
% Result: Function values so insanely large that we get infinite derivative value: we cannot proceed. 
% No step size h is small enough. This might be a problem related to numerical differentiation and Newton's method. 

end 


function [k] = test_x0(x0, restart, only_k)

methods = ["DFP", "BFGS"]; 

tols = [1, 1e-3];
k = []; 

for method=methods 
    for tol=tols        
        [kj] = run_test(x0, method, tol, restart, only_k);
        k = [k, kj]; 
    end 
end


end


function [k] = run_test(x0, method, tol, restart, only_k) 

if ~only_k
    disp("---------------------------------------------------------------------------------------")
    disp("---------------------------------------------------------------------------------------")
    fprintf("Penalty minimization:\n");
    fprintf("Using %s method\n", method);
    fprintf("Starting at point [");
    fprintf("%g, ", x0(1:end-1));
    fprintf("%g],\n", x0(end)); 
    fprintf("using tolerance %0.1e\n", tol);
    fprintf("and restart = %.0f\n", restart)

    disp("---------------------------------------------------------------------------------------")
end 
[x, k] = pen(method, x0, tol, restart);

if only_k
    k;
else  
    x
end





    

end
