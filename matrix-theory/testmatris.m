function M = testmatris(n)
% ett antal matriser att prova jordan program p?

switch n
case 0
	M = zeros(3);
case 1
	M = eye(4);
case 2
	M = [1 1 1; 1 1 1; -2 -2 -2]; 
case 3
%troligen klarar programmet inte det trots att alla egenv?rde ?r 0, men
% Matlab klanappast klarar att uppt?ka det. Ger felmeddelandet i st?llet.
	M =[-9   11 -21    63 -252;
        70 -69 141  -421 1684;
      -575 575 -1149 3451 -13801;
       3891 -3891 7782 -23345 93365;
       1024 -1024 2048 -6144 24572]; 
case 4
% given testmatris
	M = compan(poly(1:10));
case 5
% klarar ni komplexa tal?
        M = [3 -4;4 3];

case 6
 %testa hur klarar programet fel matris med olika tolerans.
	M = diag([1.000001 1]);
otherwise
	error('argumentet ska vara heltal mellan 0 och 6')
end
