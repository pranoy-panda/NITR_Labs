clc;
clearvars;
close all;

x = [0,0;0,1;1,0;1,1];%4x3
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

num_iter = 2000;
alpha = 0.1;
a = 0.1;

loss = zeros(1,num_iter);
mean_iter_loss = 0;

for i=1:num_iter
    mean_iter_loss = 0;
    for j =1:4
        %forward pass
        y1 = x(j,1)*w11 + x(j,2)*w21 + w31;
        y2 = x(j,1)*w41 + x(j,2)*w51 + w61;
        y = logsig(y1)*w12 + logsig(y2)*w22 + w32;
        y_hat = logsig(y);

        %backpropagation
        e = y_true(j,1)-y_hat
        mean_iter_loss = mean_iter_loss+e*e;
        loss_grad = e;
        
        % output layer
        grad_o = loss_grad*y_hat*(1-y_hat);       
        y_o = [y1,y2,1];
        delta_o = [delta12,delta22,delta32];
        w_o = [w12,w22,w32];
        
        delta_o = alpha*grad_o*y_o + a*delta_o;
        w_o = w_o + delta_o;
        
        w32 = w_o(3);
        w22 = w_o(2);
        w12 = w_o(1);
        
        % hidden_neuron 1
        grad_h1 = grad_o*w_o(1)*logsig(y1)*(1-logsig(y1));
        y_ = [x(j,:),1];
        delta_ = [delta11,delta21,delta31];
        w_ = [w11,w21,w31];
        [w_,delta_,grad_out] = backprop(grad_o*w_o(2),x(j,:),y2,w_,delta_,alpha,a);

        w11 = w_(3);
        w21 = w_(2);
        w31 = w_(1);
        
        % hidden_neuron 2
        grad_h2 = grad_o*w_o(2)*logsig(y2)*(1-logsig(y2));
        delta_ = [delta41,delta51,delta61];
        w_ = [w41,w51,w61];
        [w_,delta_,grad_out] = backprop(grad_o*w_o(2),x(j,:),y2,w_,delta_,alpha,a);
        w41 = w_(3);
        w51 = w_(2);
        w61 = w_(1);
        
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
xlim([1 1000]);
ylim([0 10]);