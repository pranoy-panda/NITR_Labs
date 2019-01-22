clc;
clear all;
% input matrix
x = [-1,-1,1;-1,1,1;1,-1,1;1,1,1]';% 3x4(including the bias term)
y = [-1;1;1;-1]';% 1x4 output for XOR

% weight matrix
w0 = rand(1,3); % 1x3
w1 = rand(1,2);% 1x2
%w = [1;1;1]';

%vars
num_iterations = 100;
learning_rate = 1;

for i=1:num_iterations
    %y_hat evaluation
    y_hat1 = w0*x;
    %y_hat1 = hardlims(y_hat1);
    y_hat1(2,:) = [1,1,1,1];
    y_hat2 = w1*y_hat1;
    y_hat2 = hardlims(y_hat2);
   
    e = y-y_hat2
    dL_by_dw1 = -(y-y_hat2)*y_hat1';
    dL_by_dw0 = dL_by_dw1*x(1:2,:);
    %w0 = w0+learning_rate*dL_by_dw0;
    w0 = w0+e*x';
    
end