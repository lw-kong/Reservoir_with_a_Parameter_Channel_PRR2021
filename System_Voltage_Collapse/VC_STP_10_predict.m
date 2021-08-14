
%% config
addpath '..\Functions'

warmup_r_step_cut = round( 500 /reservoir_tstep );  % drop the transient in data
warmup_r_step_length = round( 0.2 / reservoir_tstep );

predict_r_step_cut = round( 0 /reservoir_tstep );
predict_r_step_length = round( 500 / reservoir_tstep );


Q1 = 2.98983; % bifurcation parameter

tp = Q1;
Q1_warmup = min( [max(para_train_set),Q1]);

tmax_timeseries_predict = (warmup_r_step_cut + warmup_r_step_length + 5 ) * reservoir_tstep;

rng('shuffle');
tic;

%% prepare warming up data
ts_predict = NaN;
while isnan(ts_predict(end,1))
    x0 = [ 0.13*rand+0.17 ; 0.1 * rand; 0.1*rand+0.05; 0.05*rand+0.83];
    [t,ts_predict] = ode4(@(t,x) eq_VoltageCollapse(t,x,Q1_warmup),0:reservoir_tstep/ratio_tstep:tmax_timeseries_predict,x0);
end
t = t(1:ratio_tstep:end);
ts_predict = ts_predict(1:ratio_tstep:end,:);
x_warmup = ts_predict( warmup_r_step_cut+1 : warmup_r_step_cut+warmup_r_step_length, :);

%% predict
fprintf('predicting...\n');
flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
predict_r = func_STP_predict(x_warmup,tp_W * ( tp + tp_bias) ,W_in,W_r,W_out,flag_r);
toc;

%% plot
label_font_size = 12;
ticks_font_size = 12;

figure()
plot( reservoir_tstep * (0:1:length(predict_r)-1) ,predict_r(:,4))
title(['Q1 =' num2str(Q1,8)])
xlabel('t','FontSize',label_font_size)
ylabel('V','FontSize',label_font_size)
set(gca,'FontSize',ticks_font_size)
set(gcf,'color','white')

%{
figure()
plot(predict_r(700:end,1),predict_r(700:end,4))
title(['prediction of reservoir' newline 'Q1 =' num2str(Q1,8)])
set(gcf,'color','white')
%}
