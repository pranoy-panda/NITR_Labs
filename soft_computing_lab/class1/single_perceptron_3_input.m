% input matrix
x = [-1,-1,-1,1;
     -1,-1,1,1;
      -1,1,-1,1;
      -1,1,1,1 
      1,-1,-1,1;
      1,-1,1,1;
      1,1,-1,1;
      1,1,1,1 ]';% 4x8(including the bias term)
%y = [-1;-1;-1;1]';% 1x4
y = [-1;1;1;1;1;1;1;1]';% 1x8

% weight matrix
w = [0.1;0.1;-0.1;0.1]'; % 1x4

%vars
num_iterations = 6;
learning_rate = 1;

for i=1:num_iterations
    %y_hat evaluation
    y_hat = w*x ;
    y_hat = sign(y_hat);
    %error evaluation
    e = y-y_hat;
    % weight update
    w = w+e*x';
end

% 3D plot of the parameter plane
pointA = [0,0,-w(4)/w(1)];
pointB = [0,-w(4)/w(2),0];
pointC = [-w(4)/w(3),0,0];
points=[pointA' pointB' pointC'];
fill3(points(1,:),points(2,:),points(3,:),'r')
grid on
hold on;
c = x';
scatter3(c(:,1),c(:,2),c(:,3));

xlabel('X1');
ylabel('X2');
zlabel('X3');
title('parameter space');