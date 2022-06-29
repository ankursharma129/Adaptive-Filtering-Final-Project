function [R] = measurementNoiseNew(xseg,fs)
%new method of calculating measurement noise variance based on PSD

numFrame = size(xseg,1);
noise cov = zeros(1,numFrame);
spectral flatness = zeros(1,numFrame);
%order estimation for voiced and silent frames
for k = 1:numFrame

[c, lag] = xcorr(xseg(k,:),'coeff');
%calculating power spectral density from ACF
psd = (fftshift(abs(fft(c))));
psd = psd(round(length(psd)/2):end);
freq = (fs * (0:length(c)/2))/length(c);
%keeping positive lags only since ACF is symmetrical
c = c(find(lag == 0):length(c));
lag = lag(find(lag == 0):length(lag));
%keep frequencies from 100Hz to 2kHz
freq 2kHz = find(freq>= 100 & freq<=2000);
psd 2kHz = psd(freq 2kHz);
spectral flatness(k) = geomean(psd 2kHz)/mean(psd 2kHz);

end

normalized flatness = spectral flatness/max(spectral flatness);
threshold = 0.707;
for k = 1:numFrame
if normalized flatness(k) >= threshold
noise cov(k) = var(xseg(k,:));
end
end
R = max(noise cov)
end