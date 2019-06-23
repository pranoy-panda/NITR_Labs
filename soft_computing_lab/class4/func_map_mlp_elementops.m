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

w11_old = w11;
w21_old = w21;
w31_old = w31;
w41_old = w41;
w51_old = w51;
w61_old = w61;
w12_old = w12;
w22_old = w22;
w32_old = w32;

%for momentum based update
delta11 = 0;
delta21 = 0;
delta31 = 0;
delta41 = 0;
delta51 = 0;
delta61 = 0;
delta12 = 0;
delta22 = 0;
delta32 = 0;

num_iter = 5000;
alpha = 0.5;
a = 0.7;

loss = zeros(1,num_iter);
mean_iter_loss = 0;

for i=1:num_iter
    mean_iter_loss = 0;
    for j =1:4       
        
        %forward pass
        y1 = x(j,1)*w11 + x(j,2)*w21 + x(j,3)*w31;
        y2 = x(j,1)*w41 + x(j,2)*w51 + x(j,3)*w61;
        y = logsig(y1)*w12 + logsig(y2)*w22 + w32;
        y_hat = logsig(y);

        %backpropagation
        e = y_true(j,1)-y_hat
        mean_iter_loss = mean_iter_loss+e*e;
        
        % output layer
        grad_o = e*y_hat*(1-y_hat); % gradient calc.
        delta32 = alpha*grad_o + a*delta32;
        w32 = w32_old + delta32;
        delta22 = alpha*grad_o*y2 + a*delta22;
        w22 = w22_old + delta22;
        delta12 = alpha*grad_o*y1 + a*delta12;
        w12 = w12_old + delta12;
        
        % hidden_neuron 1
        grad_h1 = e*y_hat*(1-y_hat)*w12*logsig(y1)*(1-logsig(y1)); % gradient calc.
        delta11 = alpha*grad_h1*x(j,1) + a*delta11;
        w11 = w11_old + delta11;
        delta21 = alpha*grad_h1*x(j,2) + a*delta21;
        w21 = w21_old + delta21;
        delta31 = alpha*grad_h1 + a*delta31;
        w31 = w31_old + delta31;
        
        % hidden_neuron 2
        grad_h2 = e*y_hat*(1-y_hat)*w22*logsig(y2)*(1-logsig(y2)); % gradient calc.
        delta41 = alpha*grad_h2*x(j,1) + a*delta41;
        w41 = w41_old + delta41;
        delta51 = alpha*grad_h2*x(j,2) + a*delta51;
        w51 = w51_old + delta51;
        delta32 = alpha*grad_h2 + a*delta32;
        w61 = w61_old + delta32;
        
        w11_old = w11;
        w21_old = w21;
        w31_old = w31;
        w41_old = w41;
        w51_old = w51;
        w61_old = w61;
        w12_old = w12;
        w22_old = w22;
        w32_old = w32;         
    end
    loss(i) = (mean_iter_loss/4);
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

%plotting loss
figure;
plot((1:num_iter),loss);
xlim([1 num_iter]);
ylim([0 0.3]);
title('mean square error plot');
xlabel('interation number');
ylabel('MSE');