
function [u]=thomson(N,now);


  B=(now+3)/N;
  l=[1:N-1]';
  r=2*sin(pi*B.*l)./(2*pi.*l);
  rbox=[B;r];
  Ry=toeplitz(rbox);
  [u,s,v]=svd(Ry);
  u=u(:,1:now);
  v=v(:,1:now);
  s=diag(s);
  
   


