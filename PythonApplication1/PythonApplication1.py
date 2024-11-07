import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data 

def resample_time_series(data, resample_interval=0.01):
    # Resample the time data to a consistent interval (e.g., 0.01s)
    time_min = data['Time'].min()
    time_max = data['Time'].max()
    time_resampled = np.arange(time_min, time_max, resample_interval)
    
    # Interpolate PPG values to match the new time series
    ppg_resampled = np.interp(time_resampled, data['Time'], data['PPG'])
    
    resampled_data = pd.DataFrame({'Time': time_resampled, 'PPG': ppg_resampled})
    return resampled_data

def remove_noise_fft(signal, cutoff_frequency, sampling_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_signal = np.fft.fft(signal)
    filter_mask = np.abs(freqs) < cutoff_frequency
    filtered_fft = fft_signal * filter_mask
    signal = np.fft.ifft(filtered_fft).real
    return signal

def remove_noise_fourier(signal, sampling_rate):
    # Simple Fourier-based low-pass filter implementation
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_signal = np.fft.fft(signal)
    # Apply a low-pass filter with cutoff frequency of 5 Hz
    cutoff_freq = 5  # Can adjust this cutoff as needed
    filter_mask = np.abs(freqs) < cutoff_freq
    filtered_signal = np.fft.ifft(fft_signal * filter_mask).real
    return filtered_signal

def detect_peaks(data, threshold=0.1, min_distance=0.1):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            if data[i] > threshold:
                if len(peaks) == 0 or (i - peaks[-1]) > min_distance:
                    peaks.append(i)
    return peaks

def calculate_heart_rate(peaks, time_column):
    time_between_peaks = np.diff(time_column[peaks])
    average_time_per_beat = np.mean(time_between_peaks)
    heart_rate_bpm = 60 / average_time_per_beat
    return heart_rate_bpm

def plot_data_with_peaks(data, peaks, column_name):
    # Plot the filtered PPG data instead of the original
    plt.plot(data['Time'], data[column_name], label=f'{column_name} Data', alpha=0.7)
    plt.plot(data['Time'].iloc[peaks], data[column_name].iloc[peaks], "x", label='Detected Peaks', color='red')
    plt.title(f'{column_name} Data with Detected Peaks')
    plt.xlabel("Time (s)")
    plt.ylabel(column_name)
    plt.legend()
    plt.show()

# Main Procedure
if __name__ == '__main__':
    file_path = r"C:\Users\Katie\Downloads\BVP.csv"  
    sampling_rate = 256  # Define your sampling rate here (in Hz)
    data = load_data_from_csv(file_path)

    # Resample data to a consistent time interval
    data = resample_time_series(data, resample_interval=0.01)

    # Ask for the user's choice of noise removal method
    method_choice = input("Choose noise removal method (FFT/Fourier): ").strip().lower()

    if method_choice == "fft":
        data['PPG_filtered'] = remove_noise_fft(data['PPG'].values, cutoff_frequency=5, sampling_rate=sampling_rate)
    elif method_choice == "fourier":
        data['PPG_filtered'] = remove_noise_fourier(data['PPG'].values, sampling_rate)
    else:
        print("Invalid choice! Please choose either 'FFT' or 'Fourier'.")
        exit()

    # Detect peaks and calculate heart rate from filtered PPG data
    threshold_ppg = 1.0
    min_distance = 100
    ppg_peaks = detect_peaks(data['PPG_filtered'], threshold_ppg, min_distance)
    
    # Plot the data with detected peaks (from filtered data)
    plot_data_with_peaks(data, ppg_peaks, 'PPG_filtered')
    
    if len(ppg_peaks) > 1:
        heart_rate = calculate_heart_rate(ppg_peaks, data['Time'].values)
        print(f"Estimated Heart Rate from PPG: {heart_rate:.2f} BPM")
    else:
        print("Not enough peaks detected to estimate heart rate.")
