import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import os

def Muestreo(audio, sample_rate):
    audio_samples = np.array(audio.get_array_of_samples())  
    resampled_audio = np.interp(
        np.linspace(0, len(audio_samples), int(len(audio_samples) * sample_rate / audio.frame_rate)),
        np.arange(len(audio_samples)),
        audio_samples
    )
    return resampled_audio

def Cuantificacion(audio_samples, num_bits):
    quantization_levels = 2 ** num_bits
    normalized_samples = (audio_samples - audio_samples.min()) / (audio_samples.max() - audio_samples.min())
    quantized_samples = np.round(normalized_samples * (quantization_levels - 1))
    return quantized_samples

def Codificacion(quantized_samples): 
    encoded_audio = quantized_samples.astype(np.uint8).tobytes()
    return encoded_audio

def Transformacion_Fourier(audio_samples, sample_rate):
    total_Samples = len(audio_samples)
    sample_period = 1.0 / sample_rate
    yf = fft(audio_samples)
    xf = np.linspace(0.0, 1.0/(2.0*sample_period), total_Samples//2)
    return xf, 2.0/total_Samples * np.abs(yf[0:total_Samples//2])

def Histogramas(xf, yf):
    plt.subplot(2, 1, 1)
    plt.plot(xf, yf)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.title('Transformada de Fourier de la señal de audio')

    plt.subplot(2, 1, 2)
    plt.hist(yf, bins=50, color='gray', edgecolor='black')
    plt.xlabel('Amplitud')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de amplitud de las frecuencias')

    plt.tight_layout()
    plt.show()

def RemoverSilencio_y_Digitalizar(input_file, output_file, sample_rate=16000, num_bits=8, silence_thresh=-40, min_silence_len=700, keep_silence=500):
    audio = AudioSegment.from_file(input_file)

    segmentos = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)
    audioCombinado = AudioSegment.empty()
    for segmento in segmentos:
        audioCombinado += segmento

    sampled_audio = Muestreo(audioCombinado, sample_rate)
    quantized_audio = Cuantificacion(sampled_audio, num_bits)
    encoded_audio = Codificacion(quantized_audio)
    xf, yf = Transformacion_Fourier(sampled_audio, sample_rate)

    with open(output_file, 'wb') as f:
        f.write(encoded_audio)
 
    return xf, yf

input_file = 'C:/Proyecto/Audio1.mp3'
output_file = 'AudioModificado.wav'

xf, yf = RemoverSilencio_y_Digitalizar(input_file, output_file)

fundamental_freq = xf[np.argmax(yf)]
print(f"Frecuencia fundamental: {fundamental_freq} Hz")

Histogramas(xf, yf)

min_freq = xf[np.argmax(yf > 0)]
max_freq = xf[np.argmax(yf[::-1] > 0)]
print(f"Rango de frecuencia de la voz: {min_freq} Hz - {max_freq} Hz")
