
function [Ry,th1,th2,s]=thomsoncross(N,now);


  B=(now+3)/N;
  %now=fix(B*N-3)
  l=[1:N-1]';
  
  r=2*sin(pi*B.*l)./(2*pi.*l);
  rbox=[B;r];


  Ry=toeplitz(rbox);
  [u,s,v]=svd(Ry);
  s=diag(s(1:now,1:now)) 

  th1=u(:,1:now)/sqrt(now);
  th2=v(:,1:now)/sqrt(now);

   


