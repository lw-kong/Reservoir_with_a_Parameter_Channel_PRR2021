
%% config
addpath '..\Functions'

% Parameters of the food chain system
dim = 4;

% Hyperparameters of the reservoir
n = 800;
k = 250;
eig_rho = 1.6;
W_in_a = 2.1;
tp_W = 1.6;
tp_bias = -3.1;
a = 1;
beta = 1 * 10^(-4);

reservoir_tstep = 0.05;
ratio_tstep = 5;
% reservoir_tstep/ratio_tstep = length of time step for RK4

train_r_step_cut = round( 500 / reservoir_tstep ); % drop the transient in data
train_r_step_length = round( 500 /reservoir_tstep );
validate_r_step_length = round( 25 /reservoir_tstep );

bo = 5;


para_train_set = [2.98968 2.98973 2.98978];
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
        Q1 = para_train_set(tp_i);  %% system sensitive

        ts_train = NaN;
        while isnan(ts_train(end,1))
            x0 = [ 0.13*rand+0.17 ; 0.1 * rand; 0.1*rand+0.05; 0.05*rand+0.83];
            [t,ts_train] = ode4(@(t,x) eq_VoltageCollapse(t,x,Q1),0:reservoir_tstep/ratio_tstep:tmax_timeseries_train,x0);
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
    [rmse,W_in_temp,res_net_temp,P_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_train(train_data,tp_train_set,flag_r_train,1,1,1);
    fprintf('attempt rmse = %f\n',rmse)
    
    if rmse < rmse_min
        W_in = W_in_temp;
        res_net = res_net_temp;
        P = P_temp;
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
plot_dim = 4; % change the ylabel
for tp_i = 1:length(tp_train_set)
    figure('Name','Reservoir Predict');
    subplot(2,1,1)
    hold on
    plot(t_validate,x_real(tp_i,:,plot_dim));
    plot(t_validate,x_validate(tp_i,:,plot_dim),'--');
    xlabel('time');
    ylabel('V');
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


%
% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
hold on
for tp_i = 1:length(tp_train_set)
    
    
    plot(train_data(tp_i,:,1),train_data(tp_i,:,4));
    xlabel('\delta_m');
    ylabel('V');
    title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')
    
end
hold off

%