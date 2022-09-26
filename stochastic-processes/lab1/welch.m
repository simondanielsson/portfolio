function R=welch(x,dt,N,p,f,fN,window)
% R=welch(x,dt,N,p,f,fN,window)
%
% Welch spectral estimate.
%
% x      = the data
% dt     = the sampling interval
% N      = the number of segments
% p      = the segment overlap
% f      = the frequencies, must be of the form
%          0,df,2df,3df,...
% fN     = the minimum number of frequenciew to use for FFT
% window = 'rect': rectangular window, i.e.
%                  h(t)=1/sqrt(L),
%                  where  L  is the segment length.
%          'hanning': Hanning window, i.e.
%                     h(t)=(1+cos(2*pi*t/L))*sqrt(2/(3L)),
%                     where  t  is in (-L/2,L/2), and
%                            L  is the segment length.

R=zeros(size(f));
n=length(x);
N=max(1,min(n,N));
p=max(0,min(p,1));
L=n/((N-1)*(1-p)+1);
i=(0:N-1)*L*(1-p)+1;
j=i+L-1;
i=round(i);
j=round(j);
L=j-i+1;
mx=mean(x);
switch window
  case 'rect',
    for k=1:N
      xL=x(i(k):j(k));
      y=(xL-mx)/sqrt(L(k));
      Rloc=fourier(y,fN);
      Rloc=Rloc.*conj(Rloc);
      R=R+Rloc;
    end
  case 'hanning',
    for k=1:N
      xL=x(i(k):j(k));
      ii=((i(k):j(k))-(i(k)+j(k))/2)/L(k);
      h=(1+cos(2*pi*ii))*sqrt(2/3/L(k));
      y=(xL-mx).*h;
      Rloc=fourier(y,fN);
      Rloc=Rloc.*conj(Rloc);
      R=R+Rloc;
    end
  otherwise,
    disp(['Unknown window ''', window, '''.'])
end
R=R/N*dt;


function F=fourier(f,N)
P=nextpow2(2*N);
p=nextpow2(length(f));
n=2^max(P,p);
p=max(P,p)-P;
F=fft(f,n);
F=F((0:N-1)*(2^p)+1);




