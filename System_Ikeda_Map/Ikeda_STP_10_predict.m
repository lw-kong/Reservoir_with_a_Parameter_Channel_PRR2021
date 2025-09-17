addpath('..\Functions');


warmup_r_step_cut = 20;
warmup_r_step_length = 10;

predict_r_step_cut = 10;
predict_r_step_length = 2000;


% 1.0027
I_a = 1.01;
tp = I_a;
I_a_warmup = max(para_train_set);


tmax_timeseries_warmup = (warmup_r_step_cut + warmup_r_step_length + 5 ) * reservoir_tstep;

rng('shuffle');
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict
fprintf('predicting...\n');

x0 = rand+1i*rand;
x = zeros(tmax_timeseries_warmup,1);
x(1) = x0;
for t_i = 2:tmax_timeseries_warmup
    x(t_i) = func_IHJM(x(t_i-1),I_a_warmup);
end
x = x(warmup_r_step_cut+1 : warmup_r_step_cut+warmup_r_step_length);
ts_warmup = zeros(length(x),2);
ts_warmup(:,1) = real(x);
ts_warmup(:,2) = imag(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
predict_r = func_STP_predict(ts_warmup,tp_W * ( tp + tp_bias) ,W_in,res_net,P,flag_r);

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot
label_font_size = 12;
ticks_font_size = 12;

figure()
subplot(2,1,1)
plot( reservoir_tstep * (0:1:length(predict_r)-1) ,predict_r(:,1))
xlabel('t','FontSize',label_font_size)
ylabel('Re(z)','FontSize',label_font_size)
title(['a =' num2str(I_a,8)])

subplot(2,1,2)
plot( reservoir_tstep * (0:1:length(predict_r)-1) ,predict_r(:,2))
xlabel('t','FontSize',label_font_size)
ylabel('Im(z)','FontSize',label_font_size)
set(gca,'FontSize',ticks_font_size)
set(gcf,'color','white')

%
figure()
scatter(predict_r(:,1),predict_r(:,2),1)
xlabel('Re(z)','FontSize',label_font_size)
ylabel('Im(z)','FontSize',label_font_size)
title(['prediction of reservoir' newline 'a =' num2str(I_a,8)])
set(gcf,'color','white')
grid on
box on
%

