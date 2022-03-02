function [validation_performance,W_in,W_r,W_out,t_validate,x_real,x_validate] = ...
    func_STP_train(udata,tp_train_set,flag,W_in_type,W_r_type,validation_type)
% W_in: The input matrix.
% W_r: The recurrent matrix in the hidden layer, i.e. the reservoir network, or the adjacency matrix of the hidden layer.
% W_out: The output matrix.
% udata: The training data set.
% tp_train_set: The set of training control parameters.


% use multiple trials of training time series to train one W_out
% Tp is affecting globally. Each node receives the same all control parameter
% W_in_type
%           1 : Each node in the hidden layer receives all dimensions of the input, i.e. the matrix W_in is dense.
%           2 : Each node in the hidden layer receives only one dimension of the input time series. But the input control parameter is projected on all the hidden nodes.
% W_r_type
%           1 : symmeric, normally distributed, with mean 0 and variance 1
%           2 : asymmeric, uniformly distributed between 0 and 1
% validation type
%           1 : using the largest RMSE among the tp_length numbers of trials of training time series
%           2 : prediction horizon
%           3 : product of all the tp_length numbers of RMSE 
%           4 : average of all the tp_length numbers of RMSE

% size of udata: ( tp_length, temporal_steps, dim + tp_dim )

%fprintf('in train %f\n',rand)
% flag_r_train = [n k eig_rho W_in_a a beta...
%                 0 train_r_step_length validate_r_step_length reservoir_tstep dim
%                 success_threshold];
n = flag(1); % number of nodes in the hidden layer
k = flag(2); % mean degree of W_r
eig_rho = flag(3); % spectral radius of W_r
W_in_a = flag(4); % scaling of W_in
a = flag(5); % leakage
beta = flag(6); % regularization parameter of the l-2 linear regression

train_length = flag(8);
validate_length = flag(9);

tstep = flag(10); % length of each time step in the training data
dim = flag(11); % dimensionality of the training time series (excluding the control parameter)

if validation_type == 2
    success_threshold = flag(12); % tolerence threshold of error when determining prediction horizon
else
    success_threshold = 0;
end

tp_length = length(tp_train_set); % number of trials of training time series, i.e. the number of different control parameters (bifurcation parameters) in the training data
tp_dim = size(udata,3) - dim; % dimensionality of the control parameter (bifurcation parameter)

validate_start = train_length+2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define W_in
if W_in_type == 1
    W_in = W_in_a*(2*rand(n,dim+tp_dim)-1);
elseif W_in_type == 2
    % each node is inputed with with one dimenson of real data
    % and all the tuning parameters
    W_in=zeros(n,dim+tp_dim);
    n_win = n-mod(n,dim);
    index=randperm(n_win); index=reshape(index,n_win/dim,dim);
    for d_i=1:dim
        W_in(index(:,d_i),d_i)=W_in_a*(2*rand(n_win/dim,1)-1);
    end
    W_in(:,dim+1:dim+tp_dim) = W_in_a*(2*rand(n,tp_dim)-1);
elseif W_in_type == 3
    dim_sep = dim+tp_dim-1;
    W_in = zeros(n,dim_sep+1);
    n_win = n - mod(n,dim_sep);
    index = randperm(n_win); index=reshape(index,n_win/dim_sep,dim_sep);
    for d_i=1:dim_sep
        W_in(index(:,d_i),d_i) = W_in_a*(2*rand(n_win/dim_sep,1)-1);
    end
    W_in(:,end) = W_in_a*(2*rand(n,1)-1);
else
    fprintf('W_in type error\n');
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define W_r
if W_r_type == 1
    W_r=sprandsym(n,k/n); % symmeric, normally distributed, with mean 0 and variance 1.
elseif W_r_type == 2
    k = round(k);
    index1=repmat(1:n,1,k)'; % asymmeric, uniformly distributed between 0 and 1
    index2=randperm(n*k)';
    index2(:,2)=repmat(1:n,1,k)';
    index2=sortrows(index2,1);
    index1(:,2)=index2(:,2);
    W_r=sparse(index1(:,1),index1(:,2),rand(size(index1,1),1),n,n); 
else
    fprintf('res_net type error\n');
    return
end
% W_r, adjacency matrix of the hidden layer, i.e. the reservoir network
% rescale eig
eig_D=eigs(W_r,1); %only use the biggest one. Warning about the others is harmless
W_r=(eig_rho/(abs(eig_D))).*W_r;
W_r=full(W_r);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%disp('  training...')
len_washout = 10; % number of intial training steps to be dropped during regression,
% to lower the possible effects of transients in the hidden states.

r_reg = zeros(n,tp_length*(train_length-len_washout));
y_reg = zeros(dim,tp_length*(train_length-len_washout));
r_end = zeros(n,tp_length);
for tp_i = 1:tp_length % for all the different control parameters (bifurcation parameters)
    train_x = zeros(train_length,dim+tp_dim);
    train_y = zeros(train_length,dim+tp_dim);
    train_x(:,:) = udata(tp_i,1:train_length,:);
    train_y(:,:) = udata(tp_i,2:train_length+1,:); % the desired output is the prediction of the next step
    train_x = train_x';
    train_y = train_y';
    
    
    r_all = [];
    r_all(:,1) = zeros(n,1);%2*rand(n,1)-1;%
    for ti = 1:train_length
        r_all(:,ti+1) = (1-a)*r_all(:,ti) + a*tanh(W_r*r_all(:,ti)+W_in*train_x(:,ti));
    end
    r_out = r_all(:,len_washout+2:end);
    r_out(2:2:end,:) = r_out(2:2:end,:).^2; % squaring the even rows
    r_end(:,tp_i) = r_all(:,end);
    
    r_reg(:, (tp_i-1)*(train_length-len_washout) +1 : tp_i*(train_length-len_washout) ) = r_out; % the hidden state during training
    y_reg(:, (tp_i-1)*(train_length-len_washout) +1 : tp_i*(train_length-len_washout) ) = train_y(1:dim,len_washout+1:end); % the training target
end
W_out = y_reg *r_reg'*(r_reg*r_reg'+beta*eye(n))^(-1); % the ridge regression between y_reg and r_reg
% Now we got the readout layer (output layer) W_out

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% validate
%disp('validating...')
rmse_set = zeros(1,tp_length);
success_length_set = zeros(1,tp_length);

validate_predict_y_set = zeros(tp_length,validate_length,dim);
validate_real_y_set = zeros(tp_length,validate_length,dim);
for tp_i = 1:tp_length
    validate_real_y_set(tp_i,:,:) = udata(tp_i,validate_start:(validate_start+validate_length-1),1:dim);
    
    r = r_end(:,tp_i);
    u = zeros(dim+tp_dim,1); % input
    u(1:dim) = udata(tp_i,train_length+1,1:dim);
    %u(dim+1:end) = udata(tp_i,train_length+1,dim+1:end);
    for t_i = 1:validate_length
        u(dim+1:end) = udata(tp_i,train_length+t_i,dim+1:end);
        r = (1-a) * r + a * tanh(W_r*r+W_in*u);
        r_out = r;
        r_out(2:2:end,1) = r_out(2:2:end,1).^2; % squaring even rows
        predict_y = W_out * r_out; % output
        validate_predict_y_set(tp_i,t_i,:) = predict_y;
        u(1:dim) = predict_y; % update the input. The output of the current state is the input of the next step.
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    error = zeros(validate_length,dim);
    error(:,:) = validate_predict_y_set(tp_i,:,:) - validate_real_y_set(tp_i,:,:);
    %rmse_ts = sqrt( mean( abs(error).^2 ,2) );
    se_ts = sum( error.^2 ,2);
    
    success_length_set(tp_i) = validate_length * tstep;
    for t_i = 1:validate_length
        if se_ts(t_i) > success_threshold
            success_length_set(tp_i) = t_i * tstep;
            break;
        end
    end
    % if there is NaN in prediction, success length = 0 and rmse = 10
    if sum(isnan(validate_predict_y_set(:))) > 0
        success_length_set(tp_i) = 0;
        se_ts = 10;
    end
        
    rmse_set(tp_i) = sqrt(mean(se_ts));
end




if validation_type == 1
    validation_performance =  max(rmse_set);
elseif validation_type == 2
    success_length = min(success_length_set);
    %fprintf('attempt success_length = %f \n',success_length);
    validation_performance = success_length;
elseif validation_type == 3
    for tp_i = 1:tp_length
        rmse_set(tp_i) = max(rmse_set(tp_i),10^-3);
    end
    validation_performance = prod(rmse_set);
elseif validation_type == 4
    validation_performance =  mean(rmse_set);
else
    fprintf('validation type error');
    return
end


t_validate = tstep:tstep:tstep*validate_length;
x_validate = validate_predict_y_set;
x_real = validate_real_y_set;

end


