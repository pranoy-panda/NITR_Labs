clc;
clearvars;
close all;

x = [0,0,1;0,1,1;1,0,1;1,1,1];%4x3
y_true = [0,1,1,0]';%4x1

%wij -> i= index, j=layer
w11 = rand(1,1);
w21 = rand(1,1);
w31 = rand(1,1);
w41 = rand(1,1);
w51 = rand(1,1);
w61 = rand(1,1);
w12 = rand(1,1);
w22 = rand(1,1);
w32 = rand(1,1);

num_iter = 20000;
alpha = 0.07;

for i=1:num_iter
    %forward pass
    y1 = x(:,1).*w11 + x(:,2).*w21 + x(:,3).*w31;
    y2 = x(:,1).*w41 + x(:,2).*w51 + x(:,3).*w61;
    y = logsig(y1).*w12 + logsig(y2).*w22 + ones(4,1).*w32;
    y_hat = logsig(y);
    
    %backpropagation
    e = y_true-y_hat
    % output layer
    w32 = w32 + alpha*e'*(((y_hat.*(1-y_hat))).*ones(4,1));
    w22 = w22 + alpha*e'*(((y_hat.*(1-y_hat))).*y2);
    w12 = w12 + alpha*e'*(((y_hat.*(1-y_hat))).*y1);
    %hidden layer
    w11 = w11 + alpha*e'*(((y_hat.*(1-y_hat)).*w12).*((logsig(y1).*(1-logsig(y1))).*x(:,1)));
    w21 = w21 + alpha*e'*(((y_hat.*(1-y_hat)).*w12).*((logsig(y1).*(1-logsig(y1))).*x(:,2)));
    w31 = w31 + alpha*e'*(((y_hat.*(1-y_hat)).*w12).*((logsig(y1).*(1-logsig(y1))).*ones(4,1)));
    
    w41 = w41 + alpha*e'*(((logsig(y).*(1-logsig(y))).*w22).*((logsig(y2).*(1-logsig(y2))).*x(:,1)));
    w51 = w51 + alpha*e'*(((logsig(y).*(1-logsig(y))).*w22).*((logsig(y2).*(1-logsig(y2))).*x(:,2)));
    w61 = w61 + alpha*e'*(((logsig(y).*(1-logsig(y))).*w22).*((logsig(y2).*(1-logsig(y2))).*ones(4,1)));
       
end

p = [0 0 1 1; 0 1 0 1];
t = [0 1 1 0];
plotpv(p,t);
%scatter(x(1,:),x(2,:),'marker','x');
hold on;
w = [w11,w21;w41,w51]';
b = [w31,w61];
plotpc(w',b');
hold on;
xlabel('x1');
ylabel('x2'); 