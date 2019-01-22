clc;
clearvars;
close all;

x = [0,0,1;0,1,1;1,0,1;1,1,1];%4x3
y_true = [0,1,1,0]';%4x1

% w11 = -10.8663
% w21 = 11.4894
% w31 = 5.5266
% w41 = 11.7483
% w51 = -11.1213
% w61 = 5.6377
% w12 = -4.4195
% w22 = -4.3842
% w32 = 6.6022

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
alpha = 0.7;

for i=1:num_iter
    for j =1:4
        %forward pass
        y1 = x(j,1)*w11 + x(j,2)*w21 + x(j,3)*w31;
        y2 = x(j,1)*w41 + x(j,2)*w51 + x(j,3)*w61;
        y = logsig(y1)*w12 + logsig(y2)*w22 + w32;
        y_hat = logsig(y);

        %backpropagation
        e = y_true(j,1)-y_hat(1,1)
        
        w32 = w32 + alpha*e*y_hat*(1-y_hat);
        w22 = w22 + alpha*e*y_hat*(1-y_hat)*y2;
        w12 = w12 + alpha*e*y_hat*(1-y_hat)*y1;
        
        w11 = w11 + alpha*e*y_hat*(1-y_hat)*w12*(logsig(y1)*(1-logsig(y1))*x(j,1));
        w21 = w21 + alpha*e*y_hat*(1-y_hat)*w12*(logsig(y1)*(1-logsig(y1))*x(j,2));
        w31 = w31 + alpha*e*y_hat*(1-y_hat)*w12*(logsig(y1)*(1-logsig(y1)));
        
        w41 = w41 + alpha*e*y_hat*(1-y_hat)*w22*(logsig(y2)*(1-logsig(y2))*x(j,1));
        w51 = w51 + alpha*e*y_hat*(1-y_hat)*w22*(logsig(y2)*(1-logsig(y2))*x(j,2));
        w61 = w61 + alpha*e*y_hat*(1-y_hat)*w22*(logsig(y2)*(1-logsig(y2)));
        
    end
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