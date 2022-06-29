import numpy as np


def kalman(x, d, N = 64, sgm2v=1e-4):
    nIters = min(len(x),len(d)) - N
    u = np.zeros(N)
    w = np.zeros(N)
    Q = np.eye(N)*sgm2v
    P = np.eye(N)*sgm2v
    I = np.eye(N)
    e = np.zeros(nIters)
    x_hat = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        x_hat[n] = np.dot(u, w)
        e_n =  d[n] - np.dot(u, w)
        R = e_n**2+1e-10
        Pn = P + Q
        r = np.dot(Pn,u)
        K = r / (np.dot(u, r) + R + 1e-10)
        w = w + np.dot(K, e_n)
        P = np.dot(I - np.outer(K, u), Pn)
        e[n] = e_n

    return x_hat, e

import numpy as np

def rls(x, d, N = 4, lmbd = 0.999, delta = 0.01):
    nIters = min(len(x),len(d)) - N
    lmbd_inv = 1/lmbd
    u = np.zeros(N)
    w = np.zeros(N)
    P = np.eye(N)*delta
    e = np.zeros(nIters)
    x_hat = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        x_hat[n] = np.dot(u,w)
        e_n = d[n] - np.dot(u, w)
        r = np.dot(P, u)
        g = r / (lmbd + np.dot(u, r))
        w = w + e_n * g
        P = lmbd_inv*(P - np.outer(g, np.dot(u, P)))
        e[n] = e_n
    return x_hat, e

def apa(x, d, N = 4, P = 4, mu = 0.1):
    nIters = min(len(x),len(d)) - N
    u = np.zeros(N)
    A = np.zeros((N,P))
    D = np.zeros(P)
    w = np.zeros(N)
    e = np.zeros(nIters)
    x_hat = np.zeros(nIters)
    alpha = np.eye(P)*1e-2
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        A[:,1:] = A[:,:-1]
        A[:,0] = u
        D[1:] = D[:-1]
        D[0] = d[n] 
        e_n = D - np.dot(A.T, w)
        x_hat[n] = np.dot(A.T, w)[0]
        delta = np.dot(np.linalg.inv(np.dot(A.T,A)+alpha),e_n)
        w = w + mu * np.dot(A ,delta)
        e[n] = e_n[0]
    return x_hat, e

def nlms(x, d, N=4, mu=0.1):
    nIters = min(len(x),len(d)) - N
    u = np.zeros(N)
    w = np.zeros(N)
    e = np.zeros(nIters)
    x_hat = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        e_n = d[n] - np.dot(u, w)
        x_hat[n] = np.dot(u,w)
        w = w + mu * e_n * u / (np.dot(u,u)+1e-3)
        e[n] = e_n
    return x_hat, e

from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

def fdaf(x, d, M, mu=0.05, beta=0.9):
    H = np.zeros(M+1,dtype=np.complex)
    norm = np.full(M+1,1e-8)

    window =  np.hanning(M)
    x_old = np.zeros(M)

    num_block = min(len(x),len(d)) // M
    e = np.zeros(num_block*M)
    x_hat = np.zeros(num_block*M)

    for n in range(num_block):
        x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
        d_n = d[n*M:(n+1)*M]
        x_old = x[n*M:(n+1)*M]

        X_n = fft(x_n)
        y_n = ifft(H*X_n)[M:]
        e_n = d_n-y_n
        e[n*M:(n+1)*M] = e_n
        x_hat[n*M:(n+1)*M] = y_n

        e_fft = np.concatenate([np.zeros(M),e_n*window])
        E_n = fft(e_fft)

        norm = beta*norm + (1-beta)*np.abs(X_n)**2
        G = mu*E_n/(norm+1e-3)
        H = H + X_n.conj()*G

        h = ifft(H)
        h[M:] = 0
        H = fft(h)

    return x_hat, e


def fdkf(x, d, M, beta=0.95, sgm2u=1e-2, sgm2v=1e-6):
    Q = sgm2u
    R = np.full(M+1,sgm2v)
    H = np.zeros(M+1,dtype=np.complex)
    P = np.full(M+1,sgm2u)

    window =  np.hanning(M)
    x_old = np.zeros(M)

    num_block = min(len(x),len(d)) // M
    e = np.zeros(num_block*M)
    x_hat = np.zeros(num_block*M)

    for n in range(num_block):
        x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
        d_n = d[n*M:(n+1)*M]
        x_old = x[n*M:(n+1)*M]

        X_n = np.fft.rfft(x_n)

        y_n = ifft(H*X_n)[M:]
        e_n = d_n-y_n
        x_hat[n*M:(n+1)*M] = y_n

        e_fft = np.concatenate([np.zeros(M),e_n*window])
        E_n = fft(e_fft)

        R = beta*R + (1.0 - beta)*(np.abs(E_n)**2)
        P_n = P + Q*(np.abs(H))
        K = P_n*X_n.conj()/(X_n*P_n*X_n.conj()+R)
        P = (1.0 - K*X_n)*P_n 

        H = H + K*E_n
        h = ifft(H)
        h[M:] = 0
        H = fft(h)

        e[n*M:(n+1)*M] = e_n
  
    return x_hat, e

class PFDAF:
    def __init__(self, N, M, mu, partial_constrain):
        self.N = N
        self.M = M
        self.N_freq = 1+M
        self.N_fft = 2*M
        self.mu = mu
        self.partial_constrain = partial_constrain
        self.p = 0
        self.x_old = np.zeros(self.M,dtype=np.float32)
        self.X = np.zeros((N,self.N_freq),dtype=np.complex)
        self.H = np.zeros((self.N,self.N_freq),dtype=np.complex)
        self.window = np.hanning(self.M)

    def filt(self, x, d):
        assert(len(x) == self.M)
        x_now = np.concatenate([self.x_old,x])
        X = fft(x_now)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        self.x_old = x
        Y = np.sum(self.H*self.X,axis=0)
        y = ifft(Y)[self.M:]
        e = d-y
        return e, y

    def update(self,e):
        X2 = np.sum(np.abs(self.X)**2,axis=0)
        e_fft = np.zeros(shape=(self.N_fft,),dtype=np.float32)
        e_fft[self.M:] = e*self.window
        E = fft(e_fft)

        G = self.mu*E/(X2+1e-10)
        self.H += self.X.conj()*G

        if self.partial_constrain:
            h = ifft(self.H[self.p])
            h[self.M:] = 0
            self.H[self.p] = fft(h)
            self.p = (self.p + 1) % self.N
        else:
            for p in range(self.N):
                h = ifft(self.H[p])
                h[self.M:] = 0
                self.H[p] = fft(h)

def pfdaf(x, d, N=4, M=64, mu=0.2, partial_constrain=True):
    ft = PFDAF(N, M, mu, partial_constrain)
    num_block = min(len(x),len(d)) // M

    e = np.zeros(num_block*M)
    x_hat = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n, x_h = ft.filt(x_n,d_n)
        ft.update(e_n)
        e[n*M:(n+1)*M] = e_n
        x_hat[n*M:(n+1)*M] = x_h
    
    return x_hat, e


class PFDKF:
    def __init__(self,N,M,A=0.999,P_initial=1e+2, partial_constrain=True):
        self.N = N
        self.M = M
        self.N_freq = 1+M
        self.N_fft = 2*M
        self.A2 = A**2
        self.partial_constrain = partial_constrain
        self.p = 0

        self.x = np.zeros(shape=(2*self.M),dtype=np.float32)
        self.P = np.full((self.N,self.N_freq),P_initial)
        self.X = np.zeros((N,self.N_freq),dtype=np.complex)
        self.window = np.hanning(self.M)
        self.H = np.zeros((self.N,self.N_freq),dtype=np.complex)

    def filt(self, x, d):
        assert(len(x) == self.M)
        self.x[self.M:] = x
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        self.x[:self.M] = self.x[self.M:]
        Y = np.sum(self.H*self.X,axis=0)
        y = ifft(Y)[self.M:]
        e = d-y
        return e,y

    def update(self, e):
        e_fft = np.zeros(shape=(self.N_fft,),dtype=np.float32)
        e_fft[self.M:] = e*self.window
        E = fft(e_fft)
        X2 = np.sum(np.abs(self.X)**2,axis=0)
        Pe = 0.5*self.P*X2 + np.abs(E)**2/self.N
        mu = self.P / (Pe + 1e-10)
        self.P = self.A2*(1 - 0.5*mu*X2)*self.P + (1-self.A2)*np.abs(self.H)**2
        G = mu*self.X.conj()
        self.H += E*G

        if self.partial_constrain:
            h = ifft(self.H[self.p])
            h[self.M:] = 0
            self.H[self.p] = fft(h)
            self.p = (self.p + 1) % self.N
        else:
            for p in range(self.N):
                h = ifft(self.H[p])
                h[self.M:] = 0
                self.H[p] = fft(h)

def pfdkf(x, d, N=4, M=64, A=0.999,P_initial=1e+2, partial_constrain=True):
    ft = PFDKF(N, M, A, P_initial, partial_constrain)
    num_block = min(len(x),len(d)) // M

    e = np.zeros(num_block*M)
    x_hat = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n, x_h = ft.filt(x_n,d_n)
        ft.update(e_n)
        e[n*M:(n+1)*M] = e_n
        x_hat[n*M:(n+1)*M] = x_h
    return x_hat, e

def aeflaf(x, d, M=128, P=5, mu=0.2, mu_a=0.1):
    nIters = min(len(x),len(d)) - M
    Q = P*2
    u = np.zeros(M)
    w = np.zeros((Q+1)*M)
    a = 0
    e = np.zeros(nIters)
    x_hat = np.zeros(nIters)
    sk = np.zeros(P*M,dtype=np.int32)
    ck = np.zeros(P*M,dtype=np.int32)
    pk = np.tile(np.arange(P),M)
    for k in range(M):
        sk[k*P:(k+1)*P] = np.arange(1,Q,2) + k*(Q+1)
        ck[k*P:(k+1)*P] = np.arange(2,Q+1,2) + k*(Q+1)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        g = np.repeat(u,Q+1)
        g[sk] = np.exp(-a*abs(g[sk]))*np.sin(np.pi*pk*g[sk])
        g[ck] = np.exp(-a*abs(g[ck]))*np.cos(np.pi*pk*g[ck])
        y = np.dot(w, g.T)
        e_n = d[n] - y
        x_hat[n] = y
        w = w + mu*e_n*g/(np.dot(g,g)+1e-3)
        z = np.repeat(u,Q+1)
        z[sk] = -abs(z[sk])*g[sk]
        z[ck] = -abs(z[ck])*g[ck]
        z[np.arange(M)*Q] = 0
        grad_a = np.dot(z,w)
        a = a + mu_a*e_n*grad_a/(grad_a**2+1e-3)
        e[n] = e_n
    return x_hat, e


def cflaf(x, d, M=128, P=5, mu_L=0.2, mu_FL=0.5, mu_a=0.5):
    nIters = min(len(x),len(d)) - M
    Q = P*2
    beta = 0.9
    sk = np.arange(0,Q*M,2)
    ck = np.arange(1,Q*M,2)
    pk = np.tile(np.arange(P),M)
    u = np.zeros(M)
    w_L = np.zeros(M)
    w_FL = np.zeros(Q*M)
    alpha = 0
    gamma = 1
    e = np.zeros(nIters) 
    x_hat = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        g = np.repeat(u,Q)
        g[sk] = np.sin(pk*np.pi*g[sk])
        g[ck] = np.cos(pk*np.pi*g[ck])
        y_L = np.dot(w_L, u.T)
        y_FL = np.dot(w_FL,g.T)
        e_FL = d[n] - (y_L+y_FL)
        w_FL = w_FL + mu_FL * e_FL * g / (np.dot(g,g)+1e-3)
        lambda_n = 1 / (1 + np.exp(-alpha))
        y_N = y_L + lambda_n*y_FL
        e_n = d[n] - y_N
        x_hat[n] = y_N
        gamma = beta*gamma + (1-beta)*(y_FL**2)
        alpha = alpha + (mu_a*e_n*y_FL*lambda_n*(1-lambda_n) /  gamma)
        alpha = np.clip(alpha,-4,4)
        w_L = w_L + mu_L*e_n*u/(np.dot(u,u)+1e-3)
        e[n] = e_n
    return x_hat, e


def save_figs(x, x_hat, sr, filter_name):
    fig = plt.figure(figsize=(30, 20))
    plt.subplot(2,1,1)
    plt.specgram(x,Fs=sr)
    plt.title("Original noisy audio", fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.subplot(2,1,2)
    plt.specgram(x_hat,Fs=sr)
    plt.title("Filtered audio", fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig("audio/results/"+filter_name+"/10dB/plots/"+file[:-4]+"_"+noise+"_sn10"+"_filtered_spectrogram.png", dpi=50)
    plt.close(fig)

def snr(x, n):
    pass

def filter_all(d, x, sr, noise, file):
    print("Started:", noise, file)
    file_name = "cleaned_"+file[:-4]+"_"+noise+"_sn10"
    x_hat, error = kalman(x, d, N = 64, sgm2v=1e-4)
    sf.write('audio/results/kalman/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "kalman")
    snr(x_hat, error)
    
    x_hat, error = rls(x, d, N = 4, lmbd = 0.999, delta = 0.01)
    sf.write('audio/results/rls/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "rls")
    snr(x_hat, error)
    
    x_hat, error = apa(x, d, N = 4, P = 4, mu = 0.1)
    sf.write('audio/results/apa/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "apa")
    snr(x_hat, error)
    
    x_hat, error = nlms(x, d, N=4, mu=0.1)
    sf.write('audio/results/nlms/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "nlms")
    snr(x_hat, error)
    
    x_hat, error = fdaf(x, d, M=64, mu=0.05, beta=0.9)
    sf.write('audio/results/fdaf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "fdaf")
    snr(x_hat, error)
    
    x_hat, error = fdkf(x, d, M=64, beta=0.95, sgm2u=1e-2, sgm2v=1e-6)
    sf.write('audio/results/fdkf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "fdkf")
    snr(x_hat, error)
    
    x_hat, error = pfdaf(x, d, N=4, M=64, mu=0.2, partial_constrain=True)
    sf.write('audio/results/pfdaf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "pfdaf")
    snr(x_hat, error)
    
    x_hat, error = pfdkf(x, d, N=4, M=64, A=0.999,P_initial=1e+2, partial_constrain=True)
    sf.write('audio/results/pfdkf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "pfdkf")
    snr(x_hat, error)
    
    x_hat, error = aeflaf(x, d, M=128, P=5, mu=0.2, mu_a=0.1)
    sf.write('audio/results/aeflaf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "aeflaf")
    snr(x_hat, error)
    
    x_hat, error = cflaf(x, d, M=128, P=5, mu_L=0.2, mu_FL=0.5, mu_a=0.5)
    sf.write('audio/results/cflaf/10dB/wav/'+file_name+".wav", x_hat, sr, subtype='PCM_16')
    save_figs(x, x_hat, sr, "cflaf")
    snr(x_hat, error)

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import librosa
import soundfile as sf
from glob import glob
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Process
threads = []

audio_file_path = "./audio"
clean_audio = audio_file_path+"/clean"
noisy_audio_path = audio_file_path+"/noisy/allnoise_10dB"
files = [i[14:] for i in glob(clean_audio+"/*.wav")]
noise_types = ['restaurant', 'babble', 'exhibition', 'train', 'street']
for noise in noise_types:
    for file in files:
        x, sr = librosa.load(noisy_audio_path+"/"+file[:-4]+"_"+noise+"_sn10.wav")
        d, sr = librosa.load(clean_audio+"/"+file)
        threads.append(Process(target = filter_all, args=(d, x, sr, noise, file,)))

for i in threads[:10]:
    i.start()
for i in threads[:10]:
    i.join()