clc;
clear;

% 1 hidden layers, 1 output layer, 1 input layer
N = 100;% number of training eg.
M = 1;    % num. of input features
K1 = 3;   % num of neurons in 1st H layer
K2 = 3;   % num of neurons in 2nd H layer
P = 1;    % num of neurons in output layer

X = [linspace(-2*pi,2*pi,N)',ones(N,1)];%[N,M+1]
Y = sinc(linspace(-2*pi,2*pi,N)');


%plot(linspace(0,2*pi,N)',Y);
% X dim -> [N,M+1]
% generating unif. rand num in the interval [a,b]
a = 1;b = 0;
W1 = a + (b-a).*rand(M+1,K1); %[M+1,K1] plus 1 for adding bias
W2 = a + (b-a).*rand(K1+1,K2);%[K1+1,K2] plus 1 for adding bias
W3 = a + (b-a).*rand(K2+1,P); %[K2+1,P]

delta1 = zeros(M+1,K1);
delta2 = zeros(K1+1,K2);
delta3 = zeros(K2+1,P);

num_iter = 1000;
lr = 0.25;
beta = 0;

loss = 0;
% for i=1:num_iter
%     %forward pass
%     o1 = [tanh(X*W1),ones(N,1)];
%     o2 = [tanh(o1*W2),ones(N,1)];
%     y_hat = tanh(o2*W3);
%     
%     dL_by_dY_hat = Y-y_hat; % for mse loss
%     loss(i) = abs(mean(dL_by_dY_hat));
%     
%     %
%     der_act_1 = (1-tanh(o2*W3).*tanh(o2*W3));
%     flowing_grad = (dL_by_dY_hat.*der_act_1);
%     dL_by_dW3 = o2'*flowing_grad;
%     
%     %
%     der_act_2 = [(1-tanh(o1*W2).*tanh(o1*W2)),zeros(N,1)];
%     flowing_grad = (flowing_grad*W3').*der_act_2;
%     dL_by_dW2 = o1'*(flowing_grad);
%     dL_by_dW2 = dL_by_dW2(1:K1+1,1:K2);
%     
%     %
%     der_act_3 = [(1-tanh(X*W1).*tanh(X*W1)),zeros(N,1)];
%     flowing_grad = (flowing_grad(:,1:K2)*W2').*der_act_3;
%     dL_by_dW1 = X'*flowing_grad;
%     dL_by_dW1 = dL_by_dW1(1:M+1,1:K1);
%     
%     %weight update with momentum
%     delta1 = lr*dL_by_dW1*(1/N) + beta*delta1;
%     delta2 = lr*dL_by_dW2*(1/N) + beta*delta2;
%     delta3 = lr*dL_by_dW3*(1/N) + beta*delta3;
%     
%     W3 = W3 + delta3;
%     W2 = W2 + delta2;
%     W1 = W1 + delta1;    
%     
% end

for i=1:num_iter
    flowing_grad = 0;
    for j = 1:N
        %forward pass
        x = X(j,:);
        y = Y(j,:);
        o1 = [tanh(x*W1),ones(1,1)];
        o2 = [tanh(o1*W2),ones(1,1)];
        y_hat = tanh(o2*W3);

        dL_by_dY_hat = y-y_hat % for mse loss
        loss(i) = dL_by_dY_hat^2;

        %
        der_act_1 = (1-tanh(o2*W3)^2);
        flowing_grad = (dL_by_dY_hat.*der_act_1);
        dL_by_dW3 = o2'*flowing_grad;

        %
        der_act_2 = [(1-tanh(o1*W2).^2),zeros(1,1)];
        flowing_grad = (flowing_grad*W3').*der_act_2;
        dL_by_dW2 = o1'*(flowing_grad);
        dL_by_dW2 = dL_by_dW2(1:K1+1,1:K2);

        %
        der_act_3 = [(1-tanh(x*W1).^2),zeros(1,1)];
        flowing_grad = (flowing_grad(:,1:K2)*W2').*der_act_3;
        dL_by_dW1 = x'*flowing_grad;
        dL_by_dW1 = dL_by_dW1(1:M+1,1:K1);

        W3 = W3 + lr*dL_by_dW3;
        W2 = W2 + lr*dL_by_dW2;
        W1 = W1 + lr*dL_by_dW1;    
        
    end
end

o1 = [tanh(X*W1),ones(N,1)];
o2 = [tanh(o1*W2),ones(N,1)];
y_hat = tanh(o2*W3);
plot(linspace(-2*pi,2*pi,N)',y_hat);
hold on;
Y = sinc(linspace(-2*pi,2*pi,N)');
plot(linspace(-2*pi,2*pi,N)',Y,'r');
legend('estimate','True Value');

figure;
plot(linspace(1,num_iter,num_iter),loss);