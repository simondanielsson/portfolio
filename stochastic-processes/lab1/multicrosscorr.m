function [X12t,COH,H,Hang] = multicrosscorr(x1,x2,L,u12,v12,s)


[N,now]=size(u12);

if nargin<6
    s=ones(now,1);
end




for k=1:now
  X12(:,k)=s(k)*(fft(x2.*u12(:,k),L)).*conj(fft(x1.*v12(:,k),L));  
  X11(:,k)=s(k)*real(conj(fft(x1.*v12(:,k),L)).*fft(x1.*v12(:,k),L)); 
  X22(:,k)=s(k)*real(conj(fft(x2.*u12(:,k),L)).*fft(x2.*u12(:,k),L)); 
end

%plot(real(X12))
%pause


X12t=(mean((X12),2));
X11t=(mean((X11),2));
X22t=(mean((X22),2));

COH=(abs(X12t).^2)./(X11t.*X22t);
H=abs(X12t)./X11t;
Hang=angle(X12t);