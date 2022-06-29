function y = J(w)
    w0 = [-2, 2];
%     disp(w-w0);
    A = eye(2);
    d = (w-w0)'.*A;
%     disp(sum(sum(bsxfun(@times, d, (w-w0)'), 1)));
    y = sum(sum(bsxfun(@times, d, (w-w0)'), 1));
end