import datetime
import matplotlib.pyplot as plt
import os


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

def descomposar (imatge,n):
    import numpy as np
    imatge = (imatge - imatge.min())/(imatge.max()-imatge.min()) #aixi tots els valors de la imatge van entre 0 i 1
    imatge = np.floor_divide(imatge, 1/n) #ara cada valor de la imatge és la seva classe.

    return imatge