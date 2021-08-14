
%% config
addpath '..\Functions'


% Parameters of the food chain system
xc = 0.4;
yc = 2.009;
xp = 0.08;
yp = 2.876;
r0 = 0.16129;
c0 = 0.5;
dim = 3;

% Hyperparameters of the reservoir
n = 900;
k = 4;
eig_rho = 2.3;
W_in_a = 3.6;
tp_W = 0.50;
tp_bias = -2.2;
a = 0.30;
beta = 3e-5;

reservoir_tstep = 1; 
ratio_tstep = 10; 
% reservoir_tstep/ratio_tstep = length of time step for RK4

train_r_step_cut = 50 * 40 / reservoir_tstep; % drop the transient in data
train_r_step_length = 80 * 40 /reservoir_tstep;
validate_r_step_length = 30 * 40 /reservoir_tstep;
plot_dim = 3;

bo = 5; % best of


para_train_set = [0.97 0.98 0.99];
tp_train_set = para_train_set;



tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;
%% main
rmse_min = 100;
for bo_i = 1:bo
    %% preparing training data
    fprintf('preparing training data...\n');
    
    train_data_length = train_r_step_length + validate_r_step_length + 10;
    train_data = zeros(length(tp_train_set), train_data_length,dim+1); % data that goes into reservior_training
    for tp_i = 1:length(tp_train_set)
        tp = tp_train_set(tp_i);
        ghost_k = para_train_set(tp_i);  %% system sensitive
        flag_ghost = [xc yc xp yp r0 c0 ghost_k];
        ts_train = zeros(1,3);
        while min(ts_train(:,3)) < 0.5
            x0 = [ 0.4 * rand + 0.6 ; 0.4 * rand + 0.15 ; 0.5 * rand + 0.3];
            [t,ts_train] = ode4(@(t,x) eq_ghost(t,x,flag_ghost),0:reservoir_tstep/ratio_tstep:tmax_timeseries_train,x0);
        end
        t = t(1:ratio_tstep:end);
        ts_train = ts_train(1:ratio_tstep:end,:);
        ts_train = ts_train(train_r_step_cut+1:end,:); % drop the transient in data
        
        train_data(tp_i,:,1:dim) = ts_train(1:train_data_length,:);        
        train_data(tp_i,:,dim+1) = tp_W * (tp + tp_bias) * ones(train_data_length,1);        
    end
    

    %% train
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
        reservoir_tstep dim];
    [rmse,W_in_temp,W_r_temp,W_out_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_train(train_data,tp_train_set,flag_r_train,1,1,1);
    fprintf('attempt rmse = %f\n',rmse)

    
    if rmse < rmse_min
        W_in = W_in_temp;
        W_r = W_r_temp;
        W_out = W_out_temp;
        t_validate = t_validate_temp;
        x_real = x_real_temp;
        x_validate = x_validate_temp;        
        rmse_min = rmse;
    end
    
    fprintf('%f is done\n',bo_i/bo)
    toc;
end

fprintf('best rmse = %f\n',rmse_min)

%% plot
for tp_i = 1:length(tp_train_set)
    figure('Name','Reservoir Predict');
    subplot(2,1,1)
    hold on
    plot(t_validate,x_real(tp_i,:,plot_dim));
    plot(t_validate,x_validate(tp_i,:,plot_dim),'--');
    xlabel('time');
    ylabel('P');
    title(['tp = ' num2str( para_train_set(tp_i),6 )]);
    set(gcf,'color','white')
    hold off
    subplot(2,1,2)
    hold on
    plot(t_validate,abs(x_validate(tp_i,:,plot_dim)-x_real(tp_i,:,plot_dim))/...
        ( max(x_real(tp_i,:,plot_dim)) - min(x_real(tp_i,:,plot_dim)) ) )
    line([t_validate(1) t_validate(end)],[0.05 0.05])
    xlabel('Time');
    ylabel('Relative Error');
    hold off
end

