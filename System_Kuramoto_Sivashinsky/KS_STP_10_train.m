
%% config
addpath '..\Functions'

% Parameters of the KS equation
KS_N = 32;
dim = KS_N;
lyapunov_exp = 520; %%
KS_L = 1/2;
KS_v = 4;
solver_tstep = 5e-6;


% Hyperparameters of the reservoir
n = 1000; % Maybe it does not have to be so large
k = 450;
eig_rho = 0.89;
W_in_a = 0.057;
tp_W = -0.052;
tp_bias = -185;
a = 1;
beta = 8 * 10^(-5);


reservoir_tstep = 2e-5;

train_r_step_cut = ceil( 6 / reservoir_tstep ); % drop the transient in data
train_r_step_length = 12000;
validate_r_step_length = floor( 15 / lyapunov_exp /reservoir_tstep );

bo = 5; % best of

para_train_set = [196 197 198];
tp_train_set = para_train_set;


success_threshold = 8;


KS_tmax = ( train_r_step_cut + train_r_step_length + validate_r_step_length + 100 ) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;

%% preparing training data
fprintf('preparing training data...\n');
train_data = zeros(length(tp_train_set), train_r_step_length + validate_r_step_length + 8 ,dim+1); % data that goes into reservior_training
for tp_train_i = 1:length(tp_train_set)
    tp = tp_train_set(tp_train_i);
    KS_alpha = para_train_set(tp_train_i);  %% system sensitive    
    [Adata,X,~] = func_KSsim1D(KS_alpha,KS_N,KS_L,KS_v,KS_tmax,reservoir_tstep,solver_tstep);
    toc;
    
    Adata = Adata(train_r_step_cut+1:end,:);
    train_data(tp_train_i,:,1:KS_N) = Adata(1:size(train_data,2),:);    
    train_data(tp_train_i,:,dim+1) = tp_W * (tp + tp_bias) * ones(size(train_data,2),1);    %% system sensitive
end

%% train
success_length_max = 0; % success_length is the prediction horizon
for bo_i = 1:bo
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
        reservoir_tstep dim success_threshold];
    [success_length,W_in_temp,W_r_temp,W_out_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_train(train_data,tp_train_set,flag_r_train,2,2,2);
 
    
    if success_length > success_length_max
        W_in = W_in_temp;
        W_r = W_r_temp;
        W_out = W_out_temp;
        t_validate = t_validate_temp;
        x_real = x_real_temp;
        x_validate = x_validate_temp;
        success_length_max = success_length;
    end
    fprintf('attempt success length = %f \n',success_length * lyapunov_exp);
    fprintf('%f is done\n',bo_i/bo)
    toc;
end

fprintf('\n best success length is: %f\n',success_length_max * lyapunov_exp)

%% plot
y_ticks_max = 6;
color_max = 25;

xticks_set = [0 5 10 14.99];
xticks_label_set = {'0','5','10','15'};

yticks_set = [0 pi 2*pi];
yticks_label_set = {'0','\pi','2\pi'};

label_font_size = 14;
ticks_font_size = 14;

for tp_train_i = 1:length(tp_train_set)
    x_real_plot = reshape(x_real(tp_train_i,:,:),size(x_real,2),size(x_real,3));
    x_validate_plot = reshape(x_validate(tp_train_i,:,:),size(x_real,2),size(x_real,3));
    
    figure()
    subplot(3,1,1)
    hold on
    surf(lyapunov_exp * t_validate,X,x_real_plot')    
    colormap(jet)
    caxis([-color_max color_max])
    colorbar    
    view(0,90)
    shading interp
    
    %set(gca,'xtick',[])
    xticks( xticks_set );
    set(gca,'xticklabel', xticks_label_set );
    ylabel('x','FontWeight','bold','FontSize',label_font_size)    
    yticks( yticks_set );
    set(gca,'yticklabel', yticks_label_set );    
    axis([0 max(lyapunov_exp * t_validate) 0 KS_L*2*pi])
    set(gca,'FontSize',ticks_font_size);
    set(gcf,'color','white')
    title(['\alpha = ' num2str(para_train_set(tp_train_i)) ])
    hold off
    
    subplot(3,1,2)
    hold on
    surf(lyapunov_exp * t_validate,X,x_validate_plot')
    colormap(jet)
    caxis([-color_max color_max])
    colorbar
    view(0,90)
    shading interp
    
    %set(gca,'xtick',[])
    xticks( xticks_set );
    set(gca,'xticklabel', xticks_label_set );
    ylabel('x','FontWeight','bold','FontSize',label_font_size)    
    yticks( yticks_set );
    set(gca,'yticklabel', yticks_label_set );
    axis([0 max(lyapunov_exp * t_validate) 0 KS_L*2*pi])
    set(gca,'FontSize',ticks_font_size);
    set(gcf,'color','white')
    hold off
    
    subplot(3,1,3)
    hold on
    surf(lyapunov_exp * t_validate,X,(x_real_plot-x_validate_plot)')
    colormap(jet)
    caxis([-color_max color_max])
    colorbar
    view(0,90)
    shading interp
    
    xlabel('\Lambda_{m} t','FontWeight','bold','FontSize',label_font_size)
    ylabel('x','FontWeight','bold','FontSize',label_font_size)
    xticks( xticks_set );
    set(gca,'xticklabel', xticks_label_set );
    yticks( yticks_set );
    set(gca,'yticklabel', yticks_label_set );
    axis([0 max(lyapunov_exp * t_validate) 0 KS_L*2*pi])
    set(gca,'FontSize',ticks_font_size);
    set(gcf,'color','white')
    hold off
    
    figure()
    hold on
    plot(lyapunov_exp * t_validate ,sqrt( mean( abs(x_real_plot-x_validate_plot).^2 ,2) ) )
    line([0 lyapunov_exp * max(t_validate)],[success_threshold success_threshold],'LineStyle','--')
    xlabel('\Lambda_{m} t','FontWeight','bold','FontSize',label_font_size)
    ylabel('RMSE','FontWeight','bold','FontSize',label_font_size)
    set(gcf,'color','white')
    set(gca,'FontSize',ticks_font_size);
    title(['\alpha = ' num2str(para_train_set(tp_train_i)) ])
    box on
    hold off
end


