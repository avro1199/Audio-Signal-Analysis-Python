import numpy as np
import matplotlib.pyplot as plt

def manual_dft(signal):
    """
    Manually computes the DFT of a 1D signal.
    """
    N = len(signal)
    dft_result = []
    
    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = -2 * np.pi * k * n / N
            real_part += signal[n] * np.cos(angle)
            imag_part += signal[n] * np.sin(angle)
        dft_result.append(complex(real_part, imag_part))
    
    return np.array(dft_result)

def manual_fft(signal, fs, frame_size=0.02, overlap=0.5):
    """
    Analyze an audio signal using a manually implemented FFT.
    """
    frame_len = int(frame_size * fs)
    hop_size = int(frame_len * (1 - overlap))
    window = np.hanning(frame_len)

    pitch_contour = []
    harmonics_list = []
    noise_energy_list = []
    
    # Iterate through frames
    for start in range(0, len(signal) - frame_len, hop_size):
        frame = signal[start:start+frame_len] * window

        # Manual FFT Implementation
        dft_result = manual_dft(frame)
        magnitude_spectrum = np.abs(dft_result[:frame_len // 2])
        frequencies = np.linspace(0, fs / 2, frame_len // 2)

        # Step 1: Pitch Detection (Fundamental Frequency)
        peak_idx = np.argmax(magnitude_spectrum)
        f0 = frequencies[peak_idx]
        pitch_contour.append(f0)

        # Step 2: Harmonics Detection
        harmonics = []
        for n in range(2, 10):  # Check up to 10 harmonics
            harmonic_freq = n * f0
            if harmonic_freq < frequencies[-1]:
                harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                if magnitude_spectrum[harmonic_idx] > 0.1 * magnitude_spectrum[peak_idx]:  # Threshold
                    harmonics.append(harmonic_freq)
        harmonics_list.append(harmonics)

        # Step 3: Noise Estimation
        harmonic_energy = sum([magnitude_spectrum[np.argmin(np.abs(frequencies - h))] for h in harmonics])
        total_energy = sum(magnitude_spectrum)
        noise_energy = total_energy - harmonic_energy
        noise_energy_list.append(noise_energy)

    return pitch_contour, harmonics_list, noise_energy_list

# Example Usage
if __name__ == "__main__":
    from scipy.io import wavfile

    # Load an example audio file
    fs, signal = wavfile.read("pure_tone.wav")
    if signal.ndim > 1:  # Convert to mono if stereo
        signal = np.mean(signal, axis=1)

    pitch_contour, harmonics_list, noise_energy_list = manual_fft(signal, fs)

    # Plot Results
    plt.figure()
    plt.plot(pitch_contour)
    plt.title("Pitch Contour")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency (Hz)")
    plt.show()

    plt.figure()
    plt.plot(noise_energy_list)
    plt.title("Noise Energy Over Time")
    plt.xlabel("Time Frame")
    plt.ylabel("Noise Energy")
    plt.show()