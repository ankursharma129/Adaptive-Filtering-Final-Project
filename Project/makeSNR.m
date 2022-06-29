function [desiredNoise,snr] = makeSNR(x,actualNoise,dB)
%make noise of SNR 'dB' when clean signal and noise are given

%making lengths of noise and signal equal
if(length(actualNoise) > length(x))
actualNoise = actualNoise(1:length(x));
else
if(length(actualNoise) < length(x))
start = length(actualNoise)+1;
while(length(actualNoise) < length(x))
actualNoise(start:start+length(actualNoise)−1) = actualNoise(1:end);
start = length(actualNoise) + 1;
end
actualNoise = actualNoise(1:length(x));
end;
end;

sumOfSquares desired = (sum(x.ˆ2))*(10ˆ(−dB/20));
sumOfSquares given = sum(actualNoise.ˆ2);
ratio = sqrt(sumOfSquares desired/sumOfSquares given);
desiredNoise = ratio*actualNoise;
%snr should be equal to dB
snr = 20*log10(sum(x.ˆ2)/sum(desiredNoise.ˆ2));
end