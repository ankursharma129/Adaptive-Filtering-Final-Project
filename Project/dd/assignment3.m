clear;

x = -50:1:50;
y = -50:1:50;
[X, Y] = meshgrid(x,y);

W = zeros(101,101,2);

for i=1:101
    for j=1:101
        W(i,j,1) = X(i,j);
        W(i,j,2) = Y(i,j);
    end
end

Z = zeros(101,101);

for i=1:101
    for j=1:101
        Z(i,j) = J([W(i,j,1), W(i,j,2)]);
    end
end


[g,h] = gradient(Z);
contour(X,Y,Z);
hold on;
% surf(Z);
quiver(X,Y,g,h);
hold off;
% saveas(gcf, 'overlay.jpg');
mu = 0.09;
w0 = [-2, 2].';
w = [5, 15].';
p = 2;
count = 0;
while 1
    j_val = J(w);
    grad = 2*(w-w0);
    wn = w - mu*grad;
    display(wn);
    disp(norm(wn-w0));
    count = count + 1;
    if (norm(wn-w0,2)<0.001)
        break
    end
    w = wn;
%     pause(5);
end
