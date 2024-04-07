import os
import cv2
import torch
import RRDBNet_arch as arch
import numpy as np
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt


nome_modelo ="RRDB_ESRGAN_x4.pth"
dir_modelos = "models/"
caminho_da_imagem = None


def load_model(nome_modelo, dir_modelos):
    modelo_caminho = "{}{}".format(dir_modelos, nome_modelo)
    modelo = arch.RRDBNet(3,3,64,23, gc = 32)
    modelo.load_state_dict(torch.load(modelo_caminho), strict = True)
    modelo.eval()
    return modelo


img_nome = caminho_da_imagem
img = cv2.imread(caminho_da_imagem)

modelo = load_model(nome_modelo, dir_modelos)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:,:, [2,1,0]], (2,0,1))).float()
    return img.unsqueeze(0)

def postprocess_image(img_path):
    img_path = np.transpose(img_path[[2,1,0], :, :], (1,2,0))
    img_path = (img_path * 255).round().astype(np.unit8)
    return img_path



def super_resolution():
    global caminho_da_imagem
    global modelo
    if caminho_da_imagem == None:
        messagebox.showwarning('Atenção',  'Por favor, selecione um frame')
    base = os.path.splitext(os.path.basename(caminho_da_imagem))[0]
    img = cv2.imread(caminho_da_imagem)
    if img is not None:
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        LR = img.unsqueeze(0)
        with torch.no_grad():
            resultado = modelo(LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            resultado = np.transpose(resultado[[2, 1, 0], :, :], (1, 2, 0))
            resultado = (resultado * 255).round()
            cv2.imwrite('results/{:s}_EDSRGAN.png'.format(base), resultado)
            plt.imshow(resultado.astype(np.uint8))
            plt.axis('off')
            plt.title('Super Resolution EDSRGAN')
            plt.show()
        return resultado


def select_file():
   global caminho_da_imagem
   caminho_da_imagem = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.gif")])
   if caminho_da_imagem:
       caminho_da_imagem_global = caminho_da_imagem
       print(f"Arquivo selecionado: {caminho_da_imagem_global}")



def clean_esrgan():
    global caminho_da_imagem
    os.remove(caminho_da_imagem)
    messagebox.showinfo('Imagem removida', 'A imagem foi removida')



