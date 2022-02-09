function J = jordanm(A, tol)
% Authors: Simon Danielsson, F4, si7660da-s@student.lu.se

TOL = 0.1; % TODO: find a suitable default tolerance
if nargin < 2, tol = TOL; end 

J = jordanmatris(A, tol);
end

function J = jordanmatris(A, tol) 
% Calculates the jordan form of a matrix A with integer eigenvalues. Note 
% that the jordan form is unique up to permutations of the jordan blocks

%-----------------------------------------------------------------------------
% METHOD
% The method is the following:

% To compute the jordan form of A, we need to find the number of chains and
% the length of each chain, for each eigenvalue lambda_j. For every
% eigenvalue, the number of chains is exactly dim Ker (A - lambda_j I), 
% which is calculated as size(null(A - lambda_j I), 2). This is often
% called the geometric multiplicity. We note that this number is equal to
% the total number of jordan blocks for this eigenvalue. It remains to
% determine the lengths of each such chain (equal to the sizes of the
% corresponding jordan block).

% In order to determine the lengths of the chains (for each eigenvalue lambda), we
% use the following procedure. We know that a chain of at least length k is
% also of (at least) length k + 1 if and only if we can find at least one
% vector within Ker (A - lambda I)^(k+1) which is not also present in Ker
% (A - lambda I)^k (i.e. a vector in the former space complementing the basis in the latter
% space). If we can find such a vector depends on the dimensions of the
% different kernels: if the kernel increase with n dimensions between power
% j, j+1 of A - lambda I, then there must exist n chains of at least length
% j+1. This is true since one can show (easily) that a vector in Ker (A -
% lambda I)^k is also contained in Ker (A - lambda I)^(k+1). 

% Therefore, we can determine the number of chains of (at least) a specific
% length by comparing the dimensions of the kernels of subsequent powers of
% A - lambda I. We also know that we should continue with trying to find
% such linearly independent vectors until we have found an amount equal to
% that of the algebraic multiplicity of the eigenvalue lambda in question.
% Then we are done. 

% The last step is constructing the jordan blocks out of the eigenvalues
% and the chain lengths, and this is easy. If the chain is of length one, we
% simply create the 1x1 matrix ( lambda ). Else, we simply put the eigenvalue in
% the diagonal of a square matrix of size chainLength X chainLength, and
% ones on the first upper diagonal.

% Then the jordan form is obtained by putting the jordan blocks in the
% diagonal of a square block matrix of size dimA X dimA.
%-----------------------------------------------------------------------------

TOL = 0.1; % TODO: find a suitable default tolerance
if nargin < 2, tol = TOL; end 

% Fetch eigenvalues and corresponding multiplicities from input matrix A.
% If any of the eigenvalues are non-integers, an error will be thrown. 
[ev,mult] = heltalsev(A, tol);

dimA = length(A);
J = zeros(dimA); 

% Keep track of where to input the next jordan block in J. Input the next
% jordan block such that its upper left corner is located at index (pos,
% pos) in J
pos = 1; 

% Perform the algorithm for each eigenvalue individually
for i = 1:length(ev)
    lambda = ev(i); 
    I = eye(dimA);
    
    % Number of chains for this eigenvalue = geometric multiplicity
    kernelDims1 = size(null(A - lambda.*I), 2);
    
    % Allocate memory for the chain lengths and update it accordingly:
    % initially every chain has at least length = 1. 
    chainLengths = ones(kernelDims1, 1); 
    
    % Initialize the total number of chain elements aquired: we need
    % exactly one per algebraic multiplicity of the eigenvalue, i.e.
    % mult(i).
    totalChainElements = kernelDims1; 
               
    % Basis and dimension of first order of A - lambda I
    [basis, dim] = ker(A - lambda.*I);

    % Iterate through the depths/columns of the chains, start at second
    % column of elements (since the first is obtained through dim Ker (A-lambdaI). 
    depth = 2;      
    while totalChainElements < mult(i)
        % Consider the kernel of the subsequent power of A-lambdaI. We search for a
        % complementary basis to basis within nextBasis. 
        [nextBasis, nextDim] = ker(A - lambda.*I, depth);

        % We get new linearly independent vectors according to the size of
        % the complement
        numberOfChainsAtLeastLengthDepth = nextDim - dim;                        

        % Populate chain length vector: the longest chains are the ones
        % updated (i.e. at the earlier indeces) 
        for chain = 1:numberOfChainsAtLeastLengthDepth
            chainLengths(chain) = chainLengths(chain) + 1; 
        end

        % Update total number of chain elements aquired so that we know
        % when to stop looking for new linearly independent vectors. 
        totalChainElements = totalChainElements + numberOfChainsAtLeastLengthDepth; 

        % Go to next level of the chains (until we have reached enough
        % linearly independent vectors. 
        basis = nextBasis;
        dim = nextDim;            
        depth = depth + 1; 
    end
    
    % Construct the jordan blocks J__k_j(lambda_i), where k_j is the size of
    % the j'th jordan block corresponding to eigenvalue lambda_i. Evidently,
    % k_j is equal to the chain length of the j'th chain, i.e.
    % chainLengths(j). We iterate through the chain lenghs:    
    for j = 1:length(chainLengths)
        thisLength = chainLengths(j);
        J__k_j = diag(lambda.*ones(thisLength, 1)); 
        
        if (chainLengths(j) ~= 1) 
           J__k_j = J__k_j + diag(ones(thisLength - 1, 1), 1);
        end
        
        % Input the jordan block J__k_j in J at the correct position (upper
        % left corner index (pos, pos)
        J(pos:(pos + thisLength - 1), pos:(pos + thisLength - 1)) = J__k_j;
        
        % Update position for next jordan block 
        pos = pos + thisLength; 
    end
end 
end

function [basis, dim] = ker(A, power)
% [HELP FUNCTION] Returns a basis in the kernel of a matrix A^power, as well as is dimension dim

if nargin < 2, power = 1; end 

basis = null(mpower(A, power));
dim = size(basis, 2);
end

function [ev, mult] = heltalsev(A, tol)
% Checks if the matrix A has integer eigenvalues (with tolerance tol), and
% if so, computes them (with corresponding multiplicities). Else, throw an error. 

TOL = 0.1; % TODO: find a suitable tolerance
if nargin < 2, tol = TOL; end

% Estimate eigenvalues
eigVals = eig(A); 

closestInteger = round(eigVals);
isInteger = eigVals < closestInteger + tol & eigVals > closestInteger - tol;

% Check if non-integers exist
nonIntegerExists = length(find(isInteger == 0, 1)); % either 1 (yes) or 0 (does not exists)
if nonIntegerExists
    error('heltalsev:nonIntegerEigenvalues', 'Input matrix does not have integer eigenvalues (according to this given tolerance %f)', tol);
end

% Else all eigenvalues are integers.
ev = unique(closestInteger);

% Calculate multiplicities. Allocate memory for output
mult = zeros(length(ev), 1);

for i = 1:length(ev)
    mult(i) = length(find(closestInteger == ev(i)));
end

end



