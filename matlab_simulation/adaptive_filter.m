% Step1: 
   %%Time specifications:
   Fs = 8000;                   % samples per second
   dt = 1/Fs;                   % seconds per sample
   StopTime = 0.5;             % seconds
   t = (0:dt:StopTime-dt)';     % seconds

   %%Sine wave:
   Fc = 60;                     % hertz
   x = cos(2*pi*Fc*t);
   noisy_y = 0.3 * randn(length(x),1);

   d = x;
   x = x + noisy_y;
   n = length(x);

   sigma_sq = 1;
   m = 0;
   w = randi([0 1],1,8,'double');
   y = zeros(1, n);
   for i = 9:n
       sum = 0;
       w = normalize(w,'norm',1);
       for j = 1:8
           sum = sum + w(j)*x(i-j);
       end
       y(i) = sum;
       error = y(i)-d(i);
       m = (m*(i-1)+error)/i;
       sigma_sq = ((i-1)/i)*sigma_sq + (error-m)*(error-m)/(i-1);
       if (abs(error)>sigma_sq)
           for j = 1:8
               w(j) = w(j)+0.1*sign(error)*x(i-j);
           end
       else
           for j = 1:8
               w(j) = w(j) + 0.1*(2/sigma_sq - abs(error)/(sigma_sq*sigma_sq))*error*x(i-j);
           end
       end
   end

   plot(t, x);
   hold on;
%    plot(t, d);
   plot(t, y);
   legend('Original', 'Generated');