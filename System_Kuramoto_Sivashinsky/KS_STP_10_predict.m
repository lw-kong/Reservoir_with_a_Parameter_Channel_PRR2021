
%% config
addpath '..\Functions'

warmup_r_step_cut = floor( 6 / lyapunov_exp /reservoir_tstep ); % drop the transient in data
warmup_r_step_length = floor( 2 / lyapunov_exp / reservoir_tstep );

predict_r_step_cut = 0 /reservoir_tstep;
predict_r_step_length = floor( 2 / reservoir_tstep );

KS_alpha = 200.14;
tp = KS_alpha;


KS_tmax_predict = (warmup_r_step_cut + warmup_r_step_length + predict_r_step_cut + predict_r_step_length + 10)...
    * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;
%% prepare warming up data
[Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax_predict,reservoir_tstep,solver_tstep);
ts_warmup_predict = zeros(size(Adata,1),dim);
ts_warmup_predict(:,1:KS_N) = Adata;

x_warmup = ts_warmup_predict( warmup_r_step_cut  + 1 : warmup_r_step_cut + warmup_r_step_length, :);
predict_pde = ts_warmup_predict( ...
    warmup_r_step_cut + warmup_r_step_length + predict_r_step_cut + 1 :...
    warmup_r_step_cut + warmup_r_step_length + predict_r_step_cut + predict_r_step_length , :);
toc;

%% predict
fprintf('predicting...\n');
flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
predict_r = func_STP_predict(x_warmup,tp_W * ( tp + tp_bias) ,W_in,res_net,P,flag_r);
toc;

%% plot
y_ticks_max = 6;
color_max = 25;
t_predict = reservoir_tstep * (1:predict_r_step_length);

figure('Position',[300 300 950 450])
subplot(3,1,1)
hold on
surf(lyapunov_exp * t_predict,X,predict_pde')
colormap(jet)
caxis([-color_max color_max])
view(0,90)
shading interp
xlabel('\Lambda_{m} t')
ylabel('x')
colorbar
yticks( yticks_set );
set(gca,'yticklabel', yticks_label_set );
axis([0 max(lyapunov_exp * t_predict) 0 KS_L*2*pi])
set(gcf,'color','white')
title(['alpha = ' num2str(KS_alpha) ])
hold off

subplot(3,1,2)
hold on
surf(lyapunov_exp * t_predict,X,predict_r')
colormap(jet)
caxis([-color_max color_max])
view(0,90)
shading interp
xlabel('\Lambda_{m} t')
ylabel('x')
colorbar
yticks( yticks_set );
set(gca,'yticklabel', yticks_label_set );
axis([0 max(lyapunov_exp * t_predict) 0 KS_L*2*pi])
set(gcf,'color','white')
hold off

subplot(3,1,3)
hold on
surf(lyapunov_exp * t_predict,X,(predict_pde-predict_r)')
colormap(jet)
caxis([-color_max color_max])
view(0,90)
shading interp
xlabel('\Lambda_{m} t')
ylabel('x')
colorbar
yticks( yticks_set );
set(gca,'yticklabel', yticks_label_set );
axis([0 max(lyapunov_exp * t_predict) 0 KS_L*2*pi])
set(gcf,'color','white')
hold off


%
figure
hold on
plot(lyapunov_exp * t_predict , sqrt( mean( abs(predict_pde-predict_r).^2 ,2) ))
line([0 lyapunov_exp * max(t_predict)],[success_threshold success_threshold],'LineStyle','--')
xlabel('\Lambda_{m} t')
ylabel('RMSE')
set(gcf,'color','white')
title(['alpha = ' num2str(KS_alpha) ])
box on
hold off
%
