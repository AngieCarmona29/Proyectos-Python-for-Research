import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

#Paso1
def Preprocesamiento(ImgInicial, escala_grises=True, tamano=(480, 640)):
    img = cv2.imread(ImgInicial)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    if escala_grises:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, tamano)
    img = img.astype(np.float32) / 255.0
    return img

#Paso2
def MejorarContraste(img, usar_clahe=True):
    img_uint8 = (img * 255).astype(np.uint8)
    if usar_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8) / 255.0
    return cv2.equalizeHist(img_uint8) / 255.0

#Paso3a - Desenfoque Gaussiano
def ReducirRuido(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

#Paso3b
def Ruido_SalPimienta(img):
    return cv2.medianBlur((img * 255).astype(np.uint8), 3) / 255.0

#Paso4 - 
def EnfocarImagen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

#Paso5 - Evaluación
def EvaluarImagen(img_preprocesada, img_mejorada):
    img1 = (img_preprocesada * 255).astype(np.uint8)
    img2 = (img_mejorada * 255).astype(np.uint8)
    ssim_valor = ssim(img1, img2)
    psnr_valor = psnr(img1, img2)
    return ssim_valor, psnr_valor

#Paso5 - Visualización
def Resultados(inicial, Preprocesada, mejorada):
    ssim_valor, psnr_valor = EvaluarImagen(Preprocesada, mejorada)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    inicialColor=cv2.cvtColor(inicial, cv2.COLOR_BGR2RGB)

    axs[0].imshow(inicialColor)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(Preprocesada, cmap='gray')
    axs[1].set_title('Escala de grises')
    axs[1].axis('off')
    
    axs[2].imshow(mejorada, cmap='gray')
    axs[2].set_title(f'Mejorada\nSSIM: {ssim_valor:.3f}, PSNR: {psnr_valor:.3f} dB')
    axs[2].axis('off')
    plt.show()

ImgInicial = "Imagen1.jpeg"
inicial=cv2.imread(ImgInicial)
Preprocesada = Preprocesamiento(ImgInicial)
ContrasteMejorado = MejorarContraste(Preprocesada)
RuidoReducido = ReducirRuido(ContrasteMejorado)
RuidoFiltrado = Ruido_SalPimienta(RuidoReducido)
Enfocada = EnfocarImagen(RuidoFiltrado)

Resultados(inicial, Preprocesada, Enfocada)