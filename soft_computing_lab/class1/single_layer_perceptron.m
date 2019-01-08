% input matrix
x = [-1,-1,1;-1,1,1;1,-1,1;1,1,1]';% 3x4(including the bias term)
%y = [-1;-1;-1;1]';% 1x4 output for AND
y = [-1;1;1;1]';   % 1x4, output for OR

% weight matrix
w = [0.1;0.1;-0.1]'; % 1x3

%vars
num_iterations = 10;
learning_rate = 1;

for i=1:num_iterations
    %y_hat evaluation
    y_hat = w*x ;
    y_hat = hardlims(y_hat);
    %error evaluation
    e = y-y_hat;
    % weight update
    w = w+e*x';
end

plot([0,-w(3)/w(1)],[-w(3)/w(2),0]);
xlabel('X1');
ylabel('X2');
title('parameter space');
hold on;
scatter(x(1,:),x(2,:));