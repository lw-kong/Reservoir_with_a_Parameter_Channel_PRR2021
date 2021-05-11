function dxdt = eq_ghost(~,x,flag)
% 
% Dhamala, Lai. PRE 1999
% flag = [xc yc xp yp r0 c0 k];
% x(1)=r, x(3)=c, x(5)=p

xc = flag(1);
yc = flag(2);
xp = flag(3);
yp = flag(4);
r0 = flag(5);
c0 = flag(6);
k = flag(7);

dxdt = zeros(3,1);
%{
dxdt(1) = x(1) * (1 - x(1)/k) - xc * yc * x(2) * x(1)/(x(1) + r0);
dxdt(2) = xc * x(2) * (yc * x(1)/(x(1) + r0) - 1) - xp * yp * x(3) * x(2)/(x(2) + c0);
dxdt(3) = xp * x(3) * (yp * x(2)/(x(2) + c0) - 1);
%}
%{
r = x(1);
c = x(2);
p = x(3);

dxdt(1) = r * (1 - r/k) - xc*yc*c*r / (r + r0);
dxdt(2) = xc*yc*c*r / (r + r0) - xc*c  -  xp*yp*p*c / (c + c0);
dxdt(3) = xp*yp*p*c / (c + c0) - xp*p;
%}

dxdt(1) = x(1) * (1 - x(1)/k) - xc * yc * x(2) * x(1)/(x(1) + r0);
dxdt(2) = xc * x(2) * (yc * x(1)/(x(1) + r0) - 1) - xp*yp* x(3)* x(2)/(x(2) + c0);
dxdt(3) = xp * x(3) * (yp * x(2)/(x(2) + c0) - 1);

end