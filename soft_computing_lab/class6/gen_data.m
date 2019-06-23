function [ X,Y,x1,x2] = gen_data( r,d,p,N,vis )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
a = r;
b = r+d;
x1 = 0;
x2 = 0;
x1_ = 0;
x2_ = 0;

for i=1:N
    r_ = a + (b-a).*rand();
    temp = linspace(-r_,r_,N);
    x1 = [x1, temp];
    x2 = [x2, sqrt((r_)^2 - (temp).^2)];
end

if vis
    scatter(x1,x2,'filled');
    hold on;
end

for i=1:N
    r_ = a + (b-a).*rand();
    temp = linspace(r+d/2-r_,r+d/2+r_,N);
    x1_ = [x1_, temp];
    x2_ = [x2_, -real(sqrt((r_)^2 - (temp -(r+d/2)).^2)) - p];
end

if vis
    scatter(x1_,x2_,'filled');
end

x1 = [x1,x1_];
x2 = [x2,x2_];

X = [x1;x2]';
[m,~] = size(X);
X = [X,ones(m,1)];
Y = ones(floor(m/2),1);
Y = [Y; -ones(m-floor(m/2),1)];

end

