import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import irfft

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def white(N, state=None):
    state = np.random.RandomState() if state is None else state
    return state.randn(N)

def pink(N, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def blue(N, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def violet(N, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def add_noise_to_signal(signal, noise_type, noise_level=1.0, sr=10000):
    N = len(signal)
    if noise_type == 'white':
        noise = white(N, state=None) * noise_level
    elif noise_type == 'pink':
        noise = pink(N, state=None) * noise_level
    elif noise_type == 'blue':
        noise = blue(N, state=None) * noise_level
    elif noise_type == 'violet':
        noise = violet(N, state=None) * noise_level
    else:
        raise ValueError("Unsupported noise type")
    
    return signal + noise
