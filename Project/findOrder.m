function [ order ] = findOrder(noisy, dB, type, saveToPath)
%estimates order of each frame of noisy signal
totseg = size(noisy,1);
order = zeros(totseg,1);
%we assume maximum order to be 100
T = 100;

for i = 1:totseg
[arcoefs,noisevar,reflection coefs] = aryule(noisy(i,:),T);
pacf = −reflection coefs;
cpacf = cumsum(abs(pacf));
%estimated order = lag at which CPACF is 70% of range of CPACF
dist = abs(cpacf − 0.7*(range(cpacf)));
order(i) = find(dist == min(dist),1,'first');

if i == 4 | | i == totseg − 1
if i == 4
figure(5);
heading = 'PACF plot for Voiced Frame';
else
figure(6);
heading = 'PACF plot for Silent Frame';
end
title(heading);
subplot(211);
stem(pacf,'filled','MarkerSize',4);
xlabel('Lag');ylabel('Partial Autocorrelation coefficients');
xlim([1 T]);
uconf = 1.96/sqrt(size(noisy,2));
lconf = −uconf;
hold on;
plot([1 T],[1 1]'*[lconf uconf],'r');
hold off;
subplot(212);
text = ['Estimated order = ',num2str(order(i))];
stem(cpacf,'filled','MarkerSize',4);
xlabel('Lag');ylabel('Cumulative PACF');title(text);
grid on;
hold on;
plot(0.7*range(cpacf)*ones(1,T),'r');
hold off;
xlabel('Lags');ylabel('Cumulative PACF');
end
end
45
saveas(figure(5),[saveToPath,'PACF plot voiced frame ',...
type,' ',num2str(dB),'dB']);
saveas(figure(6),[saveToPath,'PACF plot silent frame ',...
type,' ',num2str(dB),'dB']);
50
end