% chose one of the best bo training results
% different tp_set and ode_parameter_set
% warm up
% no return map validation
% control parameter is 1 dim

addpath('..\Functions');


dim = 2;

% opt_Ikeda_SameData_20201104T135550_683.mat
n = 800;
k = 266;
eig_rho = 0;
W_in_a = 2.95;
tp_W = 0.456;
tp_bias = -3.00;
a = 0.84;
beta = 1.6 * 10^(-4);

% opt_Ikeda_SameData_20201104T155128_337.mat
n = 100;
k = 91;
eig_rho = 0;
W_in_a = 2;
tp_W = 0.21;
tp_bias = 2.23;
a = 1;
beta = 1 * 10^(-8);

% opt_Ikeda_SameData_20201104T151717_320.mat
n = 400;
k = 283;
eig_rho = 0.17;
W_in_a = 2.6;
tp_W = 0.35;
tp_bias = 0.47;
a = 1;
beta = 1 * 10^(-6);

%{
n = 400;
k = 100;
eig_rho = 0.2;
W_in_a = 2;
tp_W = 0.5;
tp_bias = 0.5;
a = 1;
beta = 1 * 10^(-6);
%}

reservoir_tstep = 1;
ratio_tstep = 1;

train_r_step_cut = round( 20 / reservoir_tstep );
train_r_step_length = round( 800 /reservoir_tstep );
validate_r_step_length = round( 15 /reservoir_tstep );
plot_dim = 2;
%



%validate_r_step_length = 15 /reservoir_tstep;
bo = 5;



para_train_set = [0.91 0.94 0.97];

tp_train_set = para_train_set;



tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preparing training data
rmse_min = 10000;
for bo_i = 1:bo
    fprintf('preparing training data...\n');
    
    train_data_length = train_r_step_length + validate_r_step_length + 10;
    train_data = zeros(length(tp_train_set), train_data_length,dim+1); % data that goes into reservior_training
    for tp_i = 1:length(tp_train_set)
        tp = tp_train_set(tp_i);
        I_a = para_train_set(tp_i);  %% system sensitive

        x0 = rand+1i*rand;
        x = zeros(tmax_timeseries_train,1);
        x(1) = x0;
        for t_i = 2:tmax_timeseries_train
            x(t_i,:) = func_IHJM(x(t_i-1,:),I_a);
        end
        x = x(train_r_step_cut+1:end);
        ts_train = zeros(length(x),2);
        ts_train(:,1) = real(x);
        ts_train(:,2) = imag(x);

        
        train_data(tp_i,:,1:dim) = ts_train(1:train_data_length,:);        
        train_data(tp_i,:,dim+1) = tp_W * (tp + tp_bias) * ones(train_data_length,1);    %% system sensitive        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % train
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
        reservoir_tstep dim plot_dim];
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

plot_dim = 1; % change the ylabel
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
    xlabel('time');
    ylabel('relative error');
    hold off
end

%{
plot_dim = 1; % change the ylabel
for tp_i = 1:length(tp_train_set)
    figure('Name','Reservoir Predict');
    subplot(2,1,1)
    hold on
    plot(x_real(tp_i,:,plot_dim));
    plot(x_validate(tp_i,:,plot_dim),'--');
    xlabel('time');
    ylabel('V');
    title(['tp = ' num2str( para_train_set(tp_i),6 )]);
    set(gcf,'color','white')
    hold off
    subplot(2,1,2)
    hold on
    plot(abs(x_validate(tp_i,:,plot_dim)-x_real(tp_i,:,plot_dim))/...
        ( max(x_real(tp_i,:,plot_dim)) - min(x_real(tp_i,:,plot_dim)) ) )
    line([t_validate(1) t_validate(end)],[0.05 0.05])
    xlabel('step');
    ylabel('relative error');
    hold off
end
%}

%
% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
hold on
for tp_i = 1:length(tp_train_set)
    
    
    scatter(train_data(tp_i,:,1),train_data(tp_i,:,2),1.5);
    xlabel('\delta_m');
    ylabel('V');
    title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')
    
end
hold off


%
