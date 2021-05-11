% generate KS solutions

KS_alpha = 200.14;
KS_v = 4;

KS_N = 32;
KS_L = 1/2;

KS_tmax = 3;
reservoir_tstep = 2e-5;
solver_tstep = 5e-6;


tic;
[Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax,reservoir_tstep,solver_tstep);
toc;

figure('Position',[300 300 900 250])
surf(0:reservoir_tstep:KS_tmax,X,Adata')
caxis([-25 25]);
colormap(jet)
view(0,90)
shading interp
title(['KS equation, \alpha = '  num2str(KS_alpha)])
xlabel('t'); ylabel('x')
colorbar
set(gcf,'color','white')

%figure()
%plot(0:reservoir_tstep:KS_tmax,Adata(2,:))
%plot(Adata(:,2))