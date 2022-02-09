function [] = nonlinearmin_test(printout, verbose)

% Converges 
testPosDef(printout, verbose); 

% Does not converge
testNegDef(printout, verbose); 

% Does not converge
testInDef(printout, verbose);

% Converges
testRosenbrock(printout, verbose);
testsRosenbrock(printout, verbose);

% Converges
testBooth(printout, verbose); 

% Converges if initial point is close enough to minimum [-2.9035, ...]
testStyblinskiTang(printout, verbose);  

end


function [] = testPosDef(printout, verbose)

H1 = [
    3, 0;
    0, 5
];

H3 = [
    1e6, 0;
    0, 3
];

x01 = [1; 2]; 
x02 = [-5; -3]; 

H2 = diag([1, 3, 5, 7, 9]);
x0 = [7; 7; 7; 7; 7];

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing positive definite quadratic form, min at origin")
testQuadratics(H1, x01, printout, verbose);

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing positive definite quadratic form, min at origin")
testQuadratics(H1, x02, printout, verbose);

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing positive definite quadratic form, min at origin")
testQuadratics(H3, x01, printout, verbose);

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing positive definite quadratic form, min at origin")
testQuadratics(H2, x0, printout, verbose); 

end



function [] = testNegDef(printout, verbose)

H = [
    -1, 0;
    0, -1
];
x0 = [5; 9];

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing negative definite quadratic form, has no minimum")
testQuadratics(H, x0, printout, verbose);

end



function [] = testInDef(printout, verbose)

H = [
    -3, 0;
    0, 5
];
x0 = [5; 9];

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing indefinite quadratic form, has no minimum")
testQuadratics(H, x0, printout, verbose);

end



function [] = testQuadratics(H, x0, printout, verbose)

for method=["DFP", "BFGS"] 
    try        
        testQuadratic(H, x0, method, printout, verbose)    
    catch ME 
        warning(ME.message);
    end
end

end



function [] = testQuadratic(H, x0, method, printout, verbose)

f = @(x) x'*H*x; 

[x, no_its, normg] = nonlinearmin(f, x0, method, 1e-3, 1, printout, verbose)

end



function [] = testRosenbrock(printout, verbose) 

x0 = [200; 200];

disp("--------------------------------------------------------------------------------------------")
disp("Minimizing rosenbrock function, min at (1, 1)");
disp("Initial point [" + num2str(x0(1)) + ", " + num2str(x0(2)) + "]");
[x, no_its, normg] = nonlinearmin(@rosenbrock, x0, 'DFP', 1e-6, 0, printout, verbose)

end 



function [] = testsRosenbrock(printout, verbose)

x01 = [399; -711]; 
x02 = [3990; -7111];
x03 = [200; 200];

for x0=[x01, x02, x03]
    for method=["DFP", "BFGS"]
        try
            disp("--------------------------------------------------------------------------------------------")
            disp("Minimizing rosenbrock function, min at (1, 1)");
            disp("Initial point [" + num2str(x0(1)) + ", " + num2str(x0(2)) + "]");
            disp("Using " + method + " method");
            [x, no_its, normg] = nonlinearmin(@rosenbrock, x0, method, 1e-6, 1, printout, verbose)
        catch ME
            warning(ME.warning)
        end
    end 
end

end


function [] = testBooth(printout, verbose)
% Minimum at [1; 3]

f = @(x) (x(1) + 2*x(2) - 7)^2 + (2*x(1) + x(2) - 5)^2;
x01 = [9; 10]; 
x02 = [1139; 9991];

for x0=[x01, x02]
    for method=["DFP", "BFGS"]
        try 
            disp("--------------------------------------------------------------------------------------------")
            disp("Minimizing Booth, min at (1, 3)");
            disp("Initial point [" + num2str(x0(1)) + ", " + num2str(x0(2)) + "]");
            disp("Using " + method + " method");
            [x, no_its, normg] = nonlinearmin(f, x0, method, 1e-6, 1, printout, verbose)
        catch ME
            warning(ME.message)
        end
    end
end

end 


function [] = testStyblinskiTang(printout, verbose)
% Min at approximately x=[-2.904, -2.904,..., -2.904]
% Seems to converge to minimum if starting point is sufficiently close to the optimum, independently of chosen algorithm. 

f = @(x) 1/2 * (x(1)^4 - 16*x(1)^2 + 5*x(1) + x(2)^4 - 16*x(2)^2 + 5*x(2));
x01 = [-3; -3];
x02 = [-1.5; -1.5];
x03 = [-5; -5]; 
x04 = [-1; -1]
x05 = [1; 1]

for x0=[x01, x02, x03, x04, x05]
    for method=["DFP", "BFGS"]
        disp("--------------------------------------------------------------------------------------------")
        disp("Minimizing Styblinski-Tang function, min at (-2.904, ..., -2.904)");
        disp("Initial point [" + num2str(x0(1)) + ", " + num2str(x0(2)) + "]");
        disp("Using " + method + " method");
        [x, no_its, normg] = nonlinearmin(f, x0, method, 1e-6, 1, printout, verbose)
    end
end


end