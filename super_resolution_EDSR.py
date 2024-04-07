import cv2
import os
import numpy as np
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt


caminho_da_imagem = None

def preprocess_image(caminho_da_imagem):
    image = cv2.imread(caminho_da_imagem)
    if image is None:
      raise ValueError('A leitura da imagem falhou')
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0,-1,0]])
    denoised_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return denoised_image

def combine_images_superresolution(images, upscale_factor=4, sr_model=None):
    global caminho_da_imagem
    if len(caminho_da_imagem) < 2:
        raise messagebox.showinfo('Atenção', "Pelo menos duas imagens são necessárias para super-resolução combinada.")
    sr_images = [sr_model.upsample(preprocess_image(img)) for img in caminho_da_imagem]
    superres_image = np.sum(sr_images, axis=0)
    superres_image = superres_image / len(caminho_da_imagem)
    superres_image = cv2.convertScaleAbs(superres_image)
    superres_image = cv2.bilateralFilter(superres_image, d=9, sigmaColor=75, sigmaSpace=75)
    superres_image = superres_image.astype(np.uint8)
    return superres_image


def selecionar_frames():
    global caminho_da_imagem
    caminho_da_imagem = filedialog.askopenfilenames(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.gif")])
    if caminho_da_imagem:
       caminho_da_imagem_global = caminho_da_imagem
       print(f"Arquivo selecionado: {caminho_da_imagem_global}")


def clean_edsr():
    global caminho_da_imagem
    for imagem in caminho_da_imagem:
      try:
        os.remove(imagem)
      except OSError as o:
        messagebox.showerror('Erro', 'Erro ao remover as imagens')
        return

    messagebox.showinfo('Informação', 'Imagens removidas com sucesso')



def super_resolution_EDSR():
    global caminho_da_imagem
    for i in range(1, 4):
      img_path = caminho_da_imagem
      if img_path is None:
        raise ValueError(f"A leitura da imagem {i} falhou. Verifique o caminho para o arquivo de imagem.")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/EDSR_x4.pb'
    sr.readModel(path)
    sr.setModel('edsr', 4)

    superres_image = combine_images_superresolution(caminho_da_imagem, upscale_factor=4, sr_model=sr)

    cv2.imwrite('results/imagem_super_resolvida.jpg', superres_image)

    plt.imshow(superres_image.astype(np.uint8))
    plt.axis('off')
    plt.title('Super Resolution EDSRGAN')
    plt.show()
