# Jordanisation of matrices with integral eigenvalues

The project consisted of implementing a function that computes the Jordan normal form of a given input matrix (with integral eigenvalues). The function(s) are found in <code>jordanm.m</code>.

It was made as a project in the course FMAN71 Matrix Theory at the Faculty of Engineering at Lund University. 


## How to run

Open MatLab, and <code>cd</code> the project folder. Then, simply construct your matrix <code>A</code> and input it into the function <code>jordanmatris(A, tol)</code> where tol is some tolerance level. The output will be the jordan form <code>J</code> of <code>A</code>.


## Method

The method is the following:

To compute the jordan form of A, we need to find the number of chains and
the length of each chain, for each eigenvalue lambda_j. For every
eigenvalue, the number of chains is exactly dim Ker (A - lambda_j I), 
which is calculated as size(null(A - lambda_j I), 2). This is often
called the geometric multiplicity. We note that this number is equal to
the total number of jordan blocks for this eigenvalue. It remains to
determine the lengths of each such chain (equal to the sizes of the
corresponding jordan block).

In order to determine the lengths of the chains (for each eigenvalue lambda), we
use the following procedure. We know that a chain of at least length k is
also of (at least) length k + 1 if and only if we can find at least one
vector within Ker (A - lambda I)^(k+1) which is not also present in Ker
(A - lambda I)^k (i.e. a vector in the former space complementing the basis in the latter
space). If we can find such a vector depends on the dimensions of the
different kernels: if the kernel increase with n dimensions between power
j, j+1 of A - lambda I, then there must exist n chains of at least length
j+1. This is true since one can show (easily) that a vector in Ker (A -
lambda I)^k is also contained in Ker (A - lambda I)^(k+1). 

Therefore, we can determine the number of chains of (at least) a specific
length by comparing the dimensions of the kernels of subsequent powers of
A - lambda I. We also know that we should continue with trying to find
such linearly independent vectors until we have found an amount equal to
that of the algebraic multiplicity of the eigenvalue lambda in question.
Then we are done. 

The last step is constructing the jordan blocks out of the eigenvalues
and the chain lengths, and this is easy. If the chain is of length one, we
simply create the 1x1 matrix ( lambda ). Else, we simply put the eigenvalue in
the diagonal of a square matrix of size chainLength X chainLength, and
ones on the first upper diagonal.

Then the jordan form is obtained by putting the jordan blocks in the
diagonal of a square block matrix of size dimA X dimA.
