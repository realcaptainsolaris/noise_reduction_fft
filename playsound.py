import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd

# Parameters
sampling_rate = 1000  # Hz
duration = 2  # seconds
frequencies_to_keep = [(40, 60), (110, 130)]  # Frequency ranges to preserve

# Generate time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Create a signal (50 Hz and 120 Hz sine waves)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Add random noise
noise = 2 * np.random.normal(size=t.shape)
noisy_signal = signal + noise

# Save noisy signal as WAV
write("noisy_signal.wav", sampling_rate, noisy_signal.astype(np.float32))

# Apply FFT to noisy signal
fft_values = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(fft_values), d=1 / sampling_rate)

# Create a frequency mask to filter out noise
frequency_mask = np.zeros_like(fft_values, dtype=bool)
for lower, upper in frequencies_to_keep:
    frequency_mask |= (frequencies > lower) & (frequencies < upper)

# Apply the mask to filter out unwanted frequencies
filtered_fft_values = np.where(frequency_mask, fft_values, 0)

# Perform inverse FFT to reconstruct the cleaned signal
filtered_signal = np.fft.ifft(filtered_fft_values).real

# Save the cleaned signal as WAV
write("filtered_signal.wav", sampling_rate, filtered_signal.astype(np.float32))

# Function to play audio using sounddevice


def play_audio(file_path):
    from scipy.io.wavfile import read

    sample_rate, audio = read(file_path)
    print(f"Playing {file_path}...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


# Play the original noisy signal
play_audio("noisy_signal.wav")

# Play the cleaned signal
play_audio("filtered_signal.wav")
