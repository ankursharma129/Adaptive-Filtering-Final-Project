function y = grad(w)
    A = eye(2);
    w0 = [-2, 2].';
    y = 2*A*(w-w0);
end