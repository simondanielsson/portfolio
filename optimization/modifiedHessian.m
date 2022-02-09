function [Dj_next] = modifiedHessian(f,yj, yj_next,Dj,method)
% return modified hessian matrix; 
% yj, yj_next are the points we moved between in the last step
% use grad function in directory
% method is either DFP or BFGS

pj=yj_next-yj;
qj=grad(f,yj_next)-grad(f,yj);

if strcmpi(method,'DFP')
    Dj_next=Dj+(1/(pj'*qj))*pj*pj'-(1/(qj'*Dj*qj))*Dj*qj*qj'*Dj;

elseif strcmpi(method,'BFGS')
    Dj_next=Dj+(1+(qj'*Dj*qj)/(pj'*qj))*(1/(pj'*qj))*pj*pj'-(1/(pj'*qj))*(pj*qj'*Dj+Dj*qj*pj');

else
    error(' ', 'Method type not supported')
end




