import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

def base_path(subdir=None):
    path = os.path.dirname(os.path.abspath(__file__))
    if subdir:
        path = f'{path}/{subdir}'
    return path
'''
base_path ens dóna el path del directori
'''

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
    #plt.subplot(2, 1, 1) #si les imatges són allargades millor
    plt.imshow(image_left)
    plt.subplot(1, 2, 2)
   # plt.subplot(2, 1, 2) #si les imatges són allargades millor
    plt.imshow(image_right)
    if title:
        plt.title(title)
    plt.show()
    plt.close()

'''
aquesta funció ens permet visualitzar 2 imatges una devora l'altre.
ens diu que farem una fila i dues columnes
a la primera columna colocarem image_left
a la segona columna colocarem image_right
a més si hem especificat un title el posarà
'''


def imatge_vec(imatge,n):
    dimensio = imatge.shape[0] * imatge.shape[1]
    imgvec = imatge.ravel()
    imgcol = imgvec.reshape(dimensio, n)
    return imgcol
'''
aquesta funció bàsicament el que fa es convertir una imatge en un vector. 
és a dir si tenim qualcom de l'estil:

|1|2|3|
|4|5|6|
 
ho convertirem amb: |1|2|3|4|5|6|

la n és perquè a cops a dins cada element de la matriu no tendrem un únic color sinó que en tendrem 3 rgb.
per tan al cas anteior seria n=1
però pràcticament sempre serà n=3. (n és el nombre d'elements que hi ha a cada element de l'array)
És a dir estarem en la següent situació:

|(10,2,3) |  (8,4,2) |
|(0,3,63) | (24,1,90)|
 
->|10|2|3|8|4|2|0|3|63|24|1|90|

->
|(10,2,3) |
| (8,4,2) |
|(0,3,63) |
|(24,1,90)|
 
'''


def descomposar (imatge,n):

    imatge = (imatge - imatge.min())/(imatge.max()-imatge.min()) #aixi tots els valors de la imatge van entre 0 i 1
    imatge = np.floor_divide(imatge, 1/n) #ara cada valor de la imatge és la seva classe.

    imgcol = imatge_vec(imatge,3)

    imatge = imgcol[:, 0] * n + imgcol[:, 1] + n * n * imgcol[:, 2]

    return imatge
'''
la funció descomposar el que fa és agrupar els valors dels nombres en n classes (la de 0 fins la de n-1)
el primer que feim és convertir  la imatge per tal que el rang dels valors sigui del 0 a l'1.
un pic fet això a cada valor li associam el residu de fer la divisio entera entre 1/n.

un pic fet això el que volem es que cada valor enlloc de tenir tres nombres el de R, el de G i B tenir un sol 
nombre que l'identifiqui de manera única amb aquest.

per fer aquesta operació més fàcil el que feim es convertir la imatge en una altre que té tres columnes(R,G,B)
amb la funció imatge_vec. És a dir, a imgcol tenim la imatge original però en forma de vector, és a dir, els elements
de la primera fila un davall l'altre, després el mateix amb la segona i així fins la darrera. 
Cada element de imgcol són els 3 valors rgb.

per tant ara com volem convertir-ho en un únic valor feim R*n + G +B*n^2. 
'''