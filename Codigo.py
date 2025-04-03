import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt

def apply_sampling(audio, sample_rate):
    print("Applying sampling...")
    # Convert audio to numpy array
    audio_samples = np.array(audio.get_array_of_samples())
    # Resample audio to the desired sample rate
    resampled_audio = np.interp(
        np.linspace(0, len(audio_samples), int(len(audio_samples) * sample_rate / audio.frame_rate)),
        np.arange(len(audio_samples)),
        audio_samples
    )
    return resampled_audio

def apply_quantization(audio_samples, num_bits):
    print("Applying quantization...")
    # Define the quantization levels
    quantization_levels = 2 ** num_bits
    # Normalize the samples to the range [0, 1]
    normalized_samples = (audio_samples - audio_samples.min()) / (audio_samples.max() - audio_samples.min())
    # Apply quantization
    quantized_samples = np.round(normalized_samples * (quantization_levels - 1))
    return quantized_samples

def apply_encoding(quantized_samples):
    print("Applying encoding...")
    # Encode the quantized samples to bytes
    encoded_audio = quantized_samples.astype(np.uint8).tobytes()
    return encoded_audio

def apply_fourier_transform(audio_samples, sample_rate):
    print("Applying Fourier Transform...")
    # Compute the Fourier Transform of the audio samples
    N = len(audio_samples)
    T = 1.0 / sample_rate
    yf = fft(audio_samples)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    return xf, 2.0/N * np.abs(yf[0:N//2])

def histogramas(xf, yf):
    print("Generating histograms...")
    # Histograma de frecuencia
    plt.subplot(2, 1, 1)
    plt.plot(xf, yf)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.title('Transformada de Fourier de la señal de audio')

    # Histograma de amplitud
    plt.subplot(2, 1, 2)
    plt.hist(yf, bins=50, color='gray', edgecolor='black')
    plt.xlabel('Amplitud')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de amplitud de las frecuencias')

    plt.tight_layout()
    plt.show()

def remove_silence_and_digitalize(input_file_path, output_file_path, sample_rate=16000, num_bits=8, silence_thresh=-40, min_silence_len=700, keep_silence=500):
    print("Loading audio file...")
    # Cargar el archivo de audio
    audio = AudioSegment.from_file(input_file_path)

    # Dividir el audio en segmentos eliminando los silencios
    print("Removing silence...")
    chunks = split_on_silence(audio, 
                              min_silence_len=min_silence_len,
                              silence_thresh=silence_thresh,
                              keep_silence=keep_silence)

    # Unir los segmentos de audio en un solo archivo
    combined_audio = AudioSegment.empty()
    for chunk in chunks:
        combined_audio += chunk
    # Aplicar muestreo
    sampled_audio = apply_sampling(combined_audio, sample_rate)
    # Aplicar cuantificación
    quantized_audio = apply_quantization(sampled_audio, num_bits)
    # Aplicar codificación
    encoded_audio = apply_encoding(quantized_audio)
    # Aplicar transformada de Fourier
    xf, yf = apply_fourier_transform(sampled_audio, sample_rate)
    # Guardar el archivo de audio digitalizado
    print("Saving digitalized audio file...")
    with open(output_file_path, 'wb') as f:
        f.write(encoded_audio)
    # Devolver las frecuencias y amplitudes obtenidas de la transformada de Fourier
    return xf, yf

# Ruta del archivo de entrada y salida
input_file = "Audio1.mp3"
output_file = "audio_output.wav"

# Llamar a la función para eliminar los silencios y digitalizar
xf, yf = remove_silence_and_digitalize(input_file, output_file)

# Identificar la frecuencia fundamental
fundamental_freq = xf[np.argmax(yf)]
print(f"Frecuencia fundamental: {fundamental_freq} Hz")

# Visualizar los histogramas
histogramas(xf, yf)

# Rango de frecuencia
min_freq = xf[np.argmax(yf > 0)]
max_freq = xf[np.argmax(yf[::-1] > 0)]
print(f"Rango de frecuencia de la voz: {min_freq} Hz - {max_freq} Hz")