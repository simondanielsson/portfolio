%%
a1 = -1/2;

C = 1;
A=[1 a1];


[H,w]=freqz(C,A);
R=abs(H).^2;

plot(w/2/pi,R)


%%

H=freqz(C,A,512,'whole');

Rd=abs(H).^2;

r=ifft(Rd);

stem([0:49],r(1:50))

H=freqz(C,A,512,'whole');

Rd=abs(H).^2;

r=ifft(Rd);

stem([0:49],r(1:50))

%% 
a1 = 1/2;
C= 1;
A = [1 a1];
m = 0;
n =400;
sigma = 1;

e = normrnd(m, sigma, 1, n);

x = filter(C, A, e);

%% 

clear 
clc

A=[1 -1 0.5]
C=[1 1 0.5];

P = roots(A)
Z = roots(C)

%zplane(Z,P)

[H,w]=freqz(C,A);
R=abs(H).^2;
plot(w/2/pi,R)

%% 

armagui

