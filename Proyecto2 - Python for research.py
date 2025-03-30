import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import soundfile as sf
from PIL import Image

def DeAudio_a_Espectograma(RutaAudio, RutaImgEspectograma):
    SeñalAudio, sr = librosa.load(RutaAudio)
    #Generar el espectograma
    Espectograma = librosa.feature.melspectrogram(y=SeñalAudio, sr=sr, n_mels=128)
    Espectograma_dB = librosa.power_to_db(Espectograma, ref=np.max)
    #Guardar el espectograma como imagen
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Espectograma_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectograma de Frecuencia Mel')
    plt.tight_layout()
    plt.savefig(RutaImgEspectograma)
    plt.close()

def ProcesarEspectograma (RutaImgEspectograma):
    ImgEspectograma = cv2.imread(RutaImgEspectograma)
    #Aplicación del procesamiento de imágenes
    #EspectogramaDesenfocado = cv2.GaussianBlur(ImgEspectograma, (5, 5), 0) #Desenfoque Gaussiano
    EspectogramaDesenfocado=cv2.Canny(ImgEspectograma, 100, 200)
    cv2.imwrite('EspectogramaProcesado.png', EspectogramaDesenfocado)

def Imagen_a_Matriz(EspectogramaDesenfocado):
    image = Image.open(EspectogramaDesenfocado)
    image = image.convert('L')
    image_array = np.array(image)
    return image_array

def ReconstruirAudio(EspectogramaDesenfocado, AudioReconstruido, sr=22050):
    image_array = Imagen_a_Matriz(EspectogramaDesenfocado)
    Espectograma_dB = librosa.db_to_power(image_array)
    SeñalAudio = librosa.feature.inverse.mel_to_audio(Espectograma_dB, sr=sr)
    sf.write(AudioReconstruido, SeñalAudio, sr)
    print("Audio Reconstruido")#Verificación de que se cumpla la función con normalidad

RutaAudio = 'Audio1.mp3'
RutaImgEspectograma = 'Espectrograma.png'
EspectogramaDesenfocado = 'EspectogramaProcesado.png'
AudioReconstruido = 'audio_reconstruido.wav'
DeAudio_a_Espectograma(RutaAudio, RutaImgEspectograma)
ProcesarEspectograma (RutaImgEspectograma)
Imagen_a_Matriz(EspectogramaDesenfocado)
ReconstruirAudio(EspectogramaDesenfocado, AudioReconstruido)