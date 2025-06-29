##############################################################
#                            Group: 01                       #
##############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa


def manual_dft(signal):
    """
    We are calculating DFT manually here.
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
        for n in range(2, 10):  # Checking up to 10 harmonics
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

def analyze_audio_with_spectrogram(audio_signal, sampling_rate, nfft=2048, window_size=1024, overlap=512):
    # Compute spectrogram
    frequencies, times, Sxx = spectrogram(
        audio_signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap, nfft=nfft
    )
    
    results = []
    for i in range(len(times)):
        Sxx_frame = Sxx[:, i]
        
        # Detect pitch (frequency with max magnitude)
        peak_idx = np.argmax(Sxx_frame)
        pitch = frequencies[peak_idx]
        
        # Identify harmonics
        harmonics = []
        for n in range(2, 10):  # Check up to the 10th harmonic
            harmonic_freq = n * pitch
            idx = np.argmin(np.abs(frequencies - harmonic_freq))
            if Sxx_frame[idx] > 0.1 * np.max(Sxx_frame):
                harmonics.append(frequencies[idx])
        
        # Analyze noise (spectral flatness)
        geometric_mean = np.exp(np.mean(np.log(Sxx_frame + 1e-10)))  # Avoid log(0)
        arithmetic_mean = np.mean(Sxx_frame)
        spectral_flatness = geometric_mean / arithmetic_mean
        noise_present = spectral_flatness > 0.1  # True if noise-like
        
        results.append({
            'time': times[i],
            'pitch': pitch,
            'harmonics': harmonics,
            'noise': noise_present
        })
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram with Annotations')
    plt.grid(True)

    for result in results:
        time = result['time']
        pitch = result['pitch']
        harmonics = result['harmonics']
        plt.scatter(time, pitch, color='green', label='Pitch' if time == results[0]['time'] else "")
        for harmonic in harmonics:
            plt.scatter(time, harmonic, color='blue', label='Harmonic' if time == results[0]['time'] else "")
    
    plt.legend()
    plt.show()


###################################################################################
############################ Main Code Start from Here ############################
###################################################################################

# audio_path = "full_song.wav"  # Give the audio file
# audio_path = "pure_tone.wav"  # Give the audio file
audio_path = "music.wav"  # Give the audio file

#Audio pre-processing with librosa library ## real-time feature can be added here
print("... Loading the Audio File (.wav) ...", end='')
audio_signal, sampling_rate = librosa.load(audio_path, sr=10000)
print('\tLoaded Successfully!\n\n')


print("Genereting Spectrogram using Built-in Functions\n\n")
analyze_audio_with_spectrogram(audio_signal, sampling_rate)



print("Calculating Manually......", "\t! . . Please Wait . . !", "\t\t.....It may take several seconds.....", sep='\n')
pitch_contour, harmonics_list, noise_energy_list = manual_fft(audio_signal, sampling_rate)


#Plotting from manual process
plt.figure()
plt.plot(noise_energy_list)
plt.title("Noise Energy Over Time")
plt.xlabel("Time Frame")
plt.ylabel("Noise Energy")
plt.show()

plt.figure()
plt.plot(pitch_contour)
plt.title("Pitch Contour")
plt.xlabel("Time Frame")
plt.ylabel("Frequency (Hz)")
plt.show()