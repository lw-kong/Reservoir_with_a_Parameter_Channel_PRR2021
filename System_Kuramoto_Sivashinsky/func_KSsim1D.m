function [uu,x,tt] = func_KSsim1D(alpha,N,Lx,para_v,tmax,r_step,tstep)
% modified from github.com/E-Renshaw/kuramoto-sivashinsky/blob/master/KSequ.m

%Lx = 1; % *2*pi

nplot = round(tmax/r_step);


x = Lx*2*pi*(1:N)'/N; % this is used in the plotting
u = cos(x/Lx) + 0.5 * rand(N,1); % initial condition
k = [0:N/2-1 0 -N/2+1:-1]'/Lx; % wave numbers
L = alpha * k.^2 - para_v * k.^4; % Fourier multipliers
g = -alpha * 0.5i*k;

M = 32; % no. of points for complex means
E = exp(tstep*L); 
E2 = exp(tstep*L/2);
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity
LR = tstep*L(:,ones(M,1)) + r(ones(N,1),:);
Q = tstep*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = tstep*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = tstep*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = tstep*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main loop
uu = u; tt = 0;
nmax = round(tmax/tstep); nplt = floor((tmax/nplot)/tstep);
v = fft(u);
for n = 1:nmax
    t = n*tstep;
    
    Nv = g.*fft(real(ifft(v)).^2); % N(un)
    a = E2.*v + Q.*Nv; % an
    Na = g.*fft(real(ifft(a)).^2); % N(an)
    b = E2.*v + Q.*Na; % bn
    Nb = g.*fft(real(ifft(b)).^2); % N(bn)
    c = E2.*a + Q.*(2*Nb-Nv); % cn
    Nc = g.*fft(real(ifft(c)).^2); % N(cn)
    
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        u = real(ifft(v));
        uu = [uu,u]; 
        tt = [tt,t];
    end
end

uu = uu';
end

