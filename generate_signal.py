import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
        
        window = signal.windows.gaussian(window_length, std=std)
        window = amp*window
        
        
        pos = np.random.randint(window_length, signal_length)
        
        start = pos - (window_length // 2) 
        end = pos + (window_length // 2 )
        
        window_segment = window[:end - start]  # Adjust the length of the window segment
        x[start:end] += window_segment
        '''
        plt.figure(figsize=(12, 6))
        plt.plot(t, x)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal with Gaussian Pulses')
        plt.grid(True)
        plt.show()
        '''

    return x


def gauss_sig_bipolar(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length):
    signal_length = len(x)
    
    for _ in range(num_pulses):
        std = np.random.uniform(std_min, std_max)
        amp = np.random.uniform(amp_min, amp_max)
        
        window_length = int(2 * std_length * std)  # Ensure the window length is odd
        window_length = min(window_length, signal_length)  # Ensure window length does not exceed signal length
        
        window = signal.windows.gaussian(window_length, std=std)
        window = amp * window
        
        # Alternate the sign to make it bipolar
        window = window * (-1) ** (np.arange(window_length) // (window_length // 2))
        
        # Random position for the center of the pulse
        pos = np.random.randint(window_length // 2, signal_length - window_length // 2)
        
        # Define the start and end indices for placing the window
        start = pos - (window_length // 2)
        end = pos + (window_length // 2) 
        
        # Add the window to the signal, handling boundary conditions
        x[start:end] += window[:end - start]
    
    return x



def sin_mod_base(x, t, c_freq, mod_index):
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    #prevent distortion
    mod_index = min(mod_index, 1 / np.max(np.abs(x)))
    mod_x = (1 + mod_index * x) * carrier
    return mod_x


def main():
    # Parameters
    n = 10000
    sr = 10000
    std_min = 50
    std_max = 200
    amp_min = 1
    amp_max = 20
    num_pulses = 3
    std_length = 3
    c_freq = 100
    m_index = 1

    # Build the signal
    x, t = build_empty_signal(n, sr)
    x = gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length)

    # Modulate the signal
    modulated_signal = sin_mod_base(x, t, c_freq, m_index)

    # Plot the modulated signal
    plt.figure(figsize=(12, 6))
    plt.plot(t, modulated_signal, label='Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Modulated Signal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

