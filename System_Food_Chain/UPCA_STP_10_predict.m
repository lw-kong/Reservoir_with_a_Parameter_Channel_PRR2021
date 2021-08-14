
%% config
addpath '..\Functions'

warmup_r_step_cut = 200 /reservoir_tstep; % drop the transient
warmup_r_step_length = 30 / reservoir_tstep;

predict_r_step_cut = 0 /reservoir_tstep;
predict_r_step_length = 200 * 40 / reservoir_tstep;

ghost_k = 1; % bifurcation parameter

tp = ghost_k;
ghost_k_warmup = min( [0.99,ghost_k]);

tmax_timeseries_warmup = (warmup_r_step_cut + warmup_r_step_length + 5 ) * reservoir_tstep;

rng('shuffle');
tic;

%% prepare warming up data
flag_ghost = [xc yc xp yp r0 c0 ghost_k_warmup];
ts_warmup = zeros(1,3);
while min(ts_warmup(:,3)) < 0.5
    x0 = [ 0.4 * rand + 0.6 ; 0.4 * rand + 0.15 ; 0.5 * rand + 0.3];
    [t,ts_warmup] = ode4(@(t,x) eq_ghost(t,x,flag_ghost),...
        0:reservoir_tstep/ratio_tstep:tmax_timeseries_warmup,x0);
end
t = t(1:ratio_tstep:end);
ts_warmup = ts_warmup(1:ratio_tstep:end,:);
x_warmup = ts_warmup( warmup_r_step_cut+1 : warmup_r_step_cut+warmup_r_step_length, :);

%% predict
fprintf('predicting...\n');
flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
predict_r = func_STP_predict(x_warmup,tp_W * ( tp + tp_bias) ,W_in,W_r,W_out,flag_r);
toc;

%% plot
label_font_size = 12;
ticks_font_size = 12;

plot_dim = 3;

figure()
plot( reservoir_tstep * (0:1:length(predict_r)-1) ,predict_r(:,plot_dim))
title(['k = ' num2str(ghost_k,8)])
xlabel('t','FontSize',label_font_size)
ylabel('P','FontSize',label_font_size)
set(gca,'FontSize',ticks_font_size)
set(gcf,'color','white')
