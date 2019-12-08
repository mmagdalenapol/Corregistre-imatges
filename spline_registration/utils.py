import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

def base_path(subdir=None):
    path = os.path.dirname(os.path.abspath(__file__))
    if subdir:
        path = f'{path}/{subdir}'
    return path

def get_databases_path():
    return base_path('databases')

def create_results_dir(experiment_name):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = base_path(f'results/{timestamp} {experiment_name}')
    os.makedirs(path, exist_ok=True)
    return path

def visualize_side_by_side(image_left, image_right, title=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_left)
    plt.subplot(1, 2, 2)
    plt.imshow(image_right)
    if title:
        plt.title(title)
    plt.show()
    plt.close()

def imatge_vec(imatge,n):
    dimensio = imatge.shape[0] * imatge.shape[1]
    imgvec = imatge.ravel()
    imgcol = imgvec.reshape(dimensio, n)
    return imgcol

def descomposar (imatge,n):

    imatge = (imatge - imatge.min())/(imatge.max()-imatge.min()) #aixi tots els valors de la imatge van entre 0 i 1
    imatge = np.floor_divide(imatge, 1/n) #ara cada valor de la imatge és la seva classe.

    #ara el que volem es que cada valor enlloc de tenir tres nombres el de R, el de G i B tenir un sol nombre que
    #l'identifiqui de manera única amb aquest.

    imgcol=imatge_vec(imatge,3)
    #ara a imgcol tenim la imatge original per en forma de vector, es a dir els elements de la primera fila un davall
    #l'altre,despres el mateix amb la segona i així fins la darrera. Cada element de imgcol són els 3 valors rgb

    imatge = imgcol[:, 0] * n + imgcol[:, 1] + n * n * imgcol[:, 2]

    return imatge