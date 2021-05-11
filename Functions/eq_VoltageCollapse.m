function dxdt = eq_VoltageCollapse(~,x,Q1)

deltam = x(1);
omega = x(2);
delta = x(3);
v = x(4);

Kpw = 0.4;
Kpv = 0.3;
Kqw = -0.03;
Kqv = -2.8;
Kqv2 = 2.1;
T = 8.5;
P0 = 0.6;
Q0 = 0.3;
P1 = 0;
Y0 = 3.33;
Ym = 5;
Pm = 1;
dm = 0.05;
M = 0.01464;
Em = 1.05;
theta0 = 0;
thetam = 0;

C = 3.5;
E0 = 1;
Q0 = 1.3;

theta0p = theta0 + atan( C*sin(theta0)/Y0 / (1 - C*cos(theta0/Y0)) );
Y0p = Y0 * sqrt(1+C^2/Y0^2-2*C/Y0*cos(theta0));
E0p = E0 / sqrt(1+C^2/Y0^2-2*C/Y0*cos(theta0));


P = - E0p * v * Y0p * sin(delta) + Em * v * Ym * sin(deltam-delta);
Q = E0p*v*Y0p*cos(delta) + Em*v*Ym*cos(deltam-delta) - (Y0p+Ym)*v^2;

dxdt = zeros(4,1);
dxdt(1) = omega;
dxdt(2) = ( - dm * omega + Pm - Em * v * Ym * sin( deltam - delta) ) / M;
dxdt(3) = ( -Kqv2 * v^2 - Kqv * v + Q - Q0 - Q1 ) / Kqw;
dxdt(4) = ( Kpw*Kqv2 * v^2 +(Kpw*Kqv-Kqw*Kpv) * v + Kqw * (P-P0-P1) - Kpw * (Q-Q0-Q1) ) / (T*Kqw*Kpv);

end

