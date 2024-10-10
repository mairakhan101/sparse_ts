import scipy.signal as scipy_signal
import numpy as np
import torch


def build_empty_signal(n, sr):
    t = np.arange(n) / sr
    x = np.zeros(n)
    return x, t

def gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length):
    signal_length = len(x)
    
    for _ in range(num_pulses):
        std = np.random.uniform(std_min, std_max)
        amp = np.random.uniform(amp_min, amp_max)
        
        window_length = int(2 * std_length * std)  # Ensure the window length is odd
        if window_length % 2 == 1: window_length -= 1
                
        window = scipy_signal.windows.gaussian(window_length, std=std)
        window = amp*window
        
        pos = np.random.randint(window_length, signal_length-window_length//2)
        
        start = pos - (window_length // 2) 
        end = pos + (window_length // 2 )
        
        window_segment = window[:end - start]  # Adjust the length of the window segment
        x[start:end] += window_segment

    return x, amp


import random

def generate_noise_freq(N, A, duration, sample_rate, freqs = []):
    
    t = np.linspace(0, duration, int(sample_rate*duration))  # Time axis
    
    noise = np.zeros_like(t)
    
    for freq in freqs:
        noise += A*np.sin(2*np.pi*freq*t)
        
    return noise

def generate_noise_harmonic(N, A, duration, sample_rate, freq, max_num_harmonics):
    
    t = np.linspace(0, duration, int(sample_rate*duration))  # Time axis
    num_harmonics = random.randint(2, max_num_harmonics)    # at least 2 frequencies in this harmonic
    
    noise = np.zeros_like(t)
    
    for har in range(1,num_harmonics):
        amp = random.uniform(0, A/10)
        noise += amp*np.sin(2*np.pi*freq*har*t)
        
    return noise, num_harmonics


def generate_dataset_freq(size, n, s, sr, freq, max_num_harmonics, std_min, std_max, amp_min, amp_max, num_pulses, std_length):
    X = []
    Y = []  
    Y_onehot = []   # labels that are delta functions in frequency space
    
    for i in range(size):
        x, t = build_empty_signal(n, sr)
        x, amp = gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length)
        
        noise, num_harmonics = generate_noise_harmonic(n, amp, s, sr, freq, max_num_harmonics)
        fft_signal = np.fft.fft(x)
        fft_noise = np.fft.fft(noise)
        
        inp = (abs(fft_signal+fft_noise) - min(abs(fft_signal+fft_noise))) / (max(abs(fft_signal+fft_noise)) - min(abs(fft_signal+fft_noise)))
        label = abs((fft_noise - min(fft_noise)) / (max(fft_noise) - min(fft_noise)))
        
        inp = np.fft.fftshift(inp)
        label = np.fft.fftshift(label)
        
        f = np.fft.fftfreq(n, d=1/sr)

        # label one hot
        f_shifted = np.fft.fftshift(f)
        y_onehot = [0]*n
        
        for j in range(1, num_harmonics):
            freq_idx = np.where(abs(f_shifted) == j*freq)[0]
            for f_id in freq_idx:
                y_onehot[f_id] = 1
            
        X.append(inp)
        Y.append(label) # labels that are delta functions in frequency space
        Y_onehot.append(y_onehot)   # one hot labels 
        
    return torch.tensor(X), torch.tensor(Y), torch.tensor(Y_onehot)


def generate_dataset_time(size, n, s, sr, freq, max_num_harmonics, std_min, std_max, amp_min, amp_max, num_pulses, std_length):
    X = []
    Y = []  
    # Y_onehot = []   # labels that are delta functions in frequency space
    
    for i in range(size):
        x, t = build_empty_signal(n, sr)
        x, amp = gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length)
        
        noise, num_harmonics = generate_noise_harmonic(n, amp, s, sr, freq, max_num_harmonics)
        # fft_signal = np.fft.fft(x)
        # fft_noise = np.fft.fft(noise)
        
        # inp = (abs(fft_signal+fft_noise) - min(abs(fft_signal+fft_noise))) / (max(abs(fft_signal+fft_noise)) - min(abs(fft_signal+fft_noise)))
        # label = abs((fft_noise - min(fft_noise)) / (max(fft_noise) - min(fft_noise)))
        
        # inp = np.fft.fftshift(inp)
        # label = np.fft.fftshift(label)
        
        # f = np.fft.fftfreq(n, d=1/sr)

        # label one hot
        # f_shifted = np.fft.fftshift(f)
        # y_onehot = [0]*n
        
        # for j in range(1, num_harmonics):
        #     freq_idx = np.where(abs(f_shifted) == j*freq)[0]
        #     for f_id in freq_idx:
        #         y_onehot[f_id] = 1
            
        X.append(noise+x)
        Y.append(noise) # labels that are delta functions in frequency space
        # Y_onehot.append(y_onehot)   # one hot labels 
        
    return torch.tensor(X), torch.tensor(Y)
    

    