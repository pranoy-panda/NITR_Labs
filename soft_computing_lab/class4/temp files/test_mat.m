clc;
clearvars;

% 1 hidden layers, 1 output layer, 1 input layer
N = 1000;% number of training eg.
M = 1;    % num. of input features
K1 = 3;   % num of neurons in 1st H layer
%K2 = 3;   % num of neurons in 2nd H layer
P = 1;    % num of neurons in output layer

X = [linspace(0,2*pi,N)',ones(N,1)];%[N,M+1]
Y = sin(linspace(0,2*pi,N)');
% X dim -> [N,M+1]
W1 = rand(M+1,K1); %[M+1,K1] plus 1 for adding bias
W2 = rand(K1+1,P);   %[K1+1,P] plus 1 for adding bias

num_iter = 2000;
lr = 0.5;

for i=1:num_iter
    %forward pass
    o = [logsig(X*W1),ones(N,1)];% output of hidden layer [N,K1+1]
    y_hat = logsig(o*W2);
    
    dL_by_dY_hat = Y-y_hat % for mse loss
    
    %
    der_act_1 = (logsig(o*W2)).*(1-logsig(o*W2));
    dL_by_dW2 = o'*(dL_by_dY_hat.*der_act_1);
    
    % 
    der_act_2 = [(logsig(X*W1)).*(1-logsig(X*W1)),zeros(N,1)];
    dL_by_dW1 = X'*(((dL_by_dY_hat.*der_act_1)*W2').*der_act_2);
    dL_by_dW1 = dL_by_dW1(1:M+1,1:K1);
    
    % weight update
    W1 = W1 + lr*dL_by_dW1;
    W2 = W2 + lr*dL_by_dW2;
    
    %pause(100);
end