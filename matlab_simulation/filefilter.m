[d,Fs] = audioread('test.wav');

noise = 0.01 * randn(length(d(1:1245856,1)),1);
x = d(1:1245856,1)+noise;

n = length(d(1:1245856,1));

sigma_sq = 1;
filt_size = [4, 8, 9, 10, 16];
m = 0;
y = zeros(1, n);

for l = filt_size
    w = randi([0 1],1,l,'double');
    for i = l+1:n
        sum = 0;
        w = normalize(w,'norm',1);
        for j = 1:l
           sum = sum + w(j)*x(i-j);
        end
        y(i) = sum;
        error = y(i)-d(i);
        m = (m*(i-1)+error)/i;
        sigma_sq = ((i-1)/i)*sigma_sq + (error-m)*(error-m)/(i-1);
        if (abs(error)>sigma_sq)
           for j = 1:l
               w(j) = w(j)+0.1*sign(error)*x(i-j);
           end
        else
           for j = 1:l
               w(j) = w(j) + 0.1*(2/sigma_sq - abs(error)/(sigma_sq*sigma_sq))*error*x(i-j);
           end
        end
    end
    gen = "generated"+l+".wav";
    noi = "noisy"+l+".wav";
    audiowrite(gen, y, Fs)
    audiowrite(noi, x, Fs)
end