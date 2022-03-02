function predict = func_STP_predict(x_warmup,tp,W_in,W_r,W_out,flag)
% included warmup
% flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
n = flag(1); % number of nodes in W_r
dim = flag(2); % dimensionality of the target system
a = flag(3);
warmup_length = flag(4);
predict_cut = flag(5); % drop the (possible) transient
predict_length = flag(6);

dim_tp = length(tp); % the dimensionality of the bifurcation parameter (tp). 
% In this work, we focus on the simple case where length(tp) = 1.

r = zeros(n,1); % hidden state
u = zeros(dim+dim_tp,1); % input
u(dim+1:end) = tp; % bifurcation parameter
%% warm up begin
% the purpose of warm up is to prepare the hidden state r
if warmup_length ~= 0
    x_warmup = x_warmup(1:warmup_length,:);        
    for t_i = 1:(warmup_length-1)
        u(1:dim) = x_warmup(t_i,:);
        r = (1-a) * r + a * tanh(W_r*r+W_in*u);
    end
else
    x_warmup = zeros(dim,1);
end
%% warm up end

%% predicting
% disp('  predicting...')
predict = zeros(predict_cut + predict_length,dim);
u(1:dim) = x_warmup(end,:); % prepare the starting input for prediction
% the starting hidden state r for prediction is prepared during the warm up

for t_i=1:predict_cut + predict_length
    r = (1-a) * r + a * tanh( W_r*r+W_in*u );
    
    r_out = r;
    r_out(2:2:end) = r_out(2:2:end).^2; % even number -> squared    
    predict(t_i,:) = W_out*r_out; % output
    
    u(1:dim) = predict(t_i,:); % update the input. The output of the current step becomes the input of the next step.
end

predict = predict(predict_cut+1 : end,:);


end
