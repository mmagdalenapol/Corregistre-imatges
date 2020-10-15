import datetime
import os
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
import pylab as pl


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
    # plt.subplot(2, 1, 1) #si les imatges són allargades millor
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
a més si especificam un title el posarà
'''


def imatge_vec(imatge, n):
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


def descomposar(imatge, n):

    imatge = (imatge - imatge.min()) / (imatge.max() - imatge.min())
    # aixi tots els valors de la imatge van entre 0 i 1

    imatge = np.floor_divide(imatge, 1 / n)  # ara cada valor de la imatge és la seva classe.
    imgcol = imatge_vec(imatge, 3)
    imatge = imgcol[:, 0] * n + imgcol[:, 1] + n * n * imgcol[:, 2]

    return imatge


'''
la funció descomposar el que fa és agrupar els valors dels nombres en n classes (la de 0 fins la de n-1)
el primer que feim és convertir la imatge per tal que el rang dels valors sigui del 0 a l'1.
un pic fet això a cada valor li associam el residu de fer la divisio entera entre 1/n.

un pic fet això el que volem es que cada valor enlloc de tenir tres nombres el de R, el de G i B tenir un sol 
nombre que l'identifiqui de manera única amb aquest.

per fer aquesta operació més fàcil el que feim es convertir la imatge en una altre que té tres columnes(R,G,B)
amb la funció imatge_vec. És a dir, a imgcol tenim la imatge original però en forma de vector, és a dir, els elements
de la primera fila un davall l'altre, després el mateix amb la segona i així fins la darrera. 
Cada element de imgcol són els 3 valors rgb.

per tant ara com volem convertir-ho en un únic valor feim R*n + G +B*n^2. 
'''


def coordenades_originals(imatge):
    x = np.arange(imatge.shape[0])
    y = np.arange(imatge.shape[1])

    Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
    Coord_originals_x = Coord_originals_x.ravel()
    Coord_originals_y = Coord_originals_y.ravel()

    return Coord_originals_x, Coord_originals_y


'''
x és un array de la forma [0,..., nombre de files de la imatge]
y és un array de la forma [0,..., nombre de columnes de la imatge]

feim una malla que tegui les files de x i les columnes de y. 
Guard les coordenades x de la malla a Coord_originals_x i les coordenades y a Coordenades_originals_y
Així tendrem les coordenades originals de la imatge. 
'''


def color_a_grisos(imatge):
    return 0.2125 * imatge[:, :, 0] + 0.7154 * imatge[:, :, 1] + 0.0721 * imatge[:, :, 2]


'''
és una forma de passa d'una imatge RGB a una imatge en escala de grisos
'''


def rescalar_imatge(imatge, mida_malla, pixels_per_vertex, multichannel=True):
    resolucio_ideal = (mida_malla[0] * pixels_per_vertex[0], mida_malla[1] * pixels_per_vertex[1])
    scale_factor = (resolucio_ideal[0] / imatge.shape[0], resolucio_ideal[1] / imatge.shape[1])
    imatge_rescalada = rescale(imatge, scale_factor, multichannel, anti_aliasing=False)
    return imatge_rescalada


'''
donada la mida de la malla amb la que feim feina (si tenc una malla 3 per 3 mida malla = [2,2]) 
i el nombre de pixels que volem tenir entre dos de punts de la malla calculam el factor d'escala i 
tornam la imatge reescalada que tendrà la dimensió determinada a resolucio ideal.
'''


# AMPLIAR LA MALLA
def ampliacio_malla(malla_x, malla_y):
    filx = (malla_x[0:-1, :] + malla_x[1:, :]) / 2
    fily = (malla_y[0:-1, :] + malla_y[1:, :]) / 2

    dim = [2 * malla_x.shape[0] - 1, malla_x.shape[1]]
    fil_ampl_x = np.zeros(dim)  # afegim les files entre dues de conegudes
    fil_ampl_y = np.zeros(dim)

    for i in range(0, dim[0]):
        if i % 2 == 0:  # als indexs parells deixam les files originals
            fil_ampl_x[i] = malla_x[i // 2]
            fil_ampl_y[i] = malla_y[i // 2]

        else:  # als indexs senars afegim les files interpolades
            fil_ampl_x[i] = filx[i // 2]
            fil_ampl_y[i] = fily[i // 2]

    colx = (fil_ampl_x[:, 0:-1] + fil_ampl_x[:, 1:]) / 2
    coly = (fil_ampl_y[:, 0:-1] + fil_ampl_y[:, 1:]) / 2

    dim = [2 * malla_x.shape[0] - 1, 2 * malla_x.shape[1] - 1]
    col_ampl_x = np.zeros(dim)  # afegim les columnes entre dues de conegudes
    col_ampl_y = np.zeros(dim)

    for i in range(0, dim[1]):
        if i % 2 == 0:  # als indexs parells deixam el conegut
            col_ampl_x[:, i] = fil_ampl_x[:, i // 2]
            col_ampl_y[:, i] = fil_ampl_y[:, i // 2]

        else:  # als indexs senars afegim els valors interpolats
            col_ampl_x[:, i] = colx[:, i // 2]
            col_ampl_y[:, i] = coly[:, i // 2]

    return col_ampl_x, col_ampl_y


'''
ampliacio malla ens permet passar d'una malla petita a una de més gran afegint una fila entre dues de conegudes 
i el mateix amb les columnes interpolant els valors d'aquestes noves entrades a partir dels valors coneguts.
'''




def visualitza_malla(imatge, malla_x, malla_y, title, path_guardar=None):
    # podem fer tant la malla inicial com l'òptima
    plt.axis('off')
    plt.imshow(imatge)
    plt.scatter(malla_y,malla_x,label='skitscat', color='k', s=25,marker="o")
    plt.title(title)
    if path_guardar:
        pl.savefig(path_guardar)
    plt.close()

