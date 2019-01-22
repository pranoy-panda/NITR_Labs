% % input matrix
%x = [2,2,1;1,-2,1;-2,2,1;-1,1,1]';% 3x4(including the bias term)
% %y = [-1;-1;-1;1]';% 1x4 output for AND
%y = [-1;1;-1;1]';   % 1x4, output for OR
% input matrix
x = [-1,-1,1;-1,1,1;1,-1,1;1,1,1]';% 3x4(including the bias term)
y = [-1;-1;-1;1]';% 1x4 output for AND
%y = [-1;1;1;1]';   % 1x4, output for OR

% weight matrix
w = rand(1,3); % 1x3
w2 = rand(1,3);
%w = [1;1;1]';

%vars
num_iterations = 1000;
learning_rate = 0.01;

for i=1:num_iterations
    %y_hat evaluation
    y_hat = w*x ;
    y_hat = purelin(y_hat);
    y_hat2 = w2*x;
    y_hat2 = hardlims(y_hat2);
    %y_hat = ~(w*x<0);
    %error evaluation
    e = y-y_hat;
    e2 = y-y_hat2;
    % weight update
    w = w+learning_rate*e*x';
    w2 = w2+e2*x';
    
    %plotting decision boundary
    %scatter(x(1:2,:),x(2,:),'marker','x');
%     plotpv(x(1:2,:),y);
%     hold on;
%     plotpc(w(1:2),w(3));
%     xlabel('x1');
%     ylabel('x2');   
%     
%     pause(1);
end
    %scatter(x(1,:),x(2,:),'marker','x');   
    hPlot = plotpc(w(1:2),w(3));
    set(hPlot,'Color','r');
    hold on;
    xlabel('x1');
    ylabel('x2'); 
    plotpc(w2(1:2),w2(3));
    legend({'Hardlims','Adalin'});
    hold on;
    y = y>0;
    plotpv(x(1:2,:),y);    

    %plot([0,-w2(3)/w2(1)],[-w2(3)/w2(2),0],'Color','r');
    
   

