
import numpy as np
from skimage.io import imread, imsave
from time import time
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab as pl
from skimage.filters import gaussian
from skimage.transform import rescale
import random
from spline_registration.transform_models import ElasticTransform
from spline_registration.utils import coordenades_originals, create_results_dir,rescalar_imatge, color_a_grisos,ampliacio_malla
from spline_registration.losses import RMSE, info_mutua
from scipy.optimize import least_squares, minimize

def main():
    random.seed(2020)
    n=2
    mida_malla1 = [n, n]
    pixels_per_vertex = 15

    # IMATGES ORIGINLAS
    imatge_input_orig = imread('dog_input.jpg')
    imatge_ref_orig = imread('dog_reference.jpg')
    dim_original = imatge_input_orig.shape

    # IMATGES RESCALADES
    imatge_input = rescalar_imatge(imatge_input_orig, mida_malla1, pixels_per_vertex)
    imatge_reference = rescalar_imatge(imatge_ref_orig, mida_malla1, pixels_per_vertex)
    dim_imatge = imatge_input.shape
    # objecte de la classe ElasticTransform amb la dimensió de la malla inicial i la dimensió de la imatge reduida.
    corregistre1 = ElasticTransform(mida_malla1, dim_imatge)

    #imatges amb les que faig feina:
    input = color_a_grisos(imatge_input)
    reference = color_a_grisos(imatge_reference)
    input_orig = color_a_grisos(imatge_input_orig)
    reference_orig = color_a_grisos(imatge_ref_orig)
    multichannel = False
    gamma=10

    path_carpeta_experiment = create_results_dir(f'diverses malles inicials')
    # millor malla 3,3
    malla_vector1 = corregistre1.malla_inicial()
    millor_malla_preliminar1 = guardar_millor_imatge_registrada(corregistre1,input, reference, malla_vector1,
                                                                         path_carpeta_experiment, 100, 1/4,
                                                                      gamma, None)
    for malla in millor_malla_preliminar1:
        malla_ampliada = ampliacio_malla(malla[0], malla[1])
        malla_vector = [malla_ampliada[0].ravel(), malla_ampliada[1].ravel()]
        mida_malla2 = [2 * mida_malla1[0], 2 * mida_malla1[1]]
        input = rescalar_imatge(input_orig, mida_malla2, pixels_per_vertex - 5, multichannel)
        reference = rescalar_imatge(reference_orig, mida_malla2, pixels_per_vertex - 5, multichannel)
        dim_imatge2 = input.shape

        parametres_redimensionats = np.concatenate([dim_imatge2[0] / dim_imatge[0] * malla_vector[0],
                                                    dim_imatge2[1] / dim_imatge[1] * malla_vector[1]])
        corregistre2 = ElasticTransform(mida_malla2, dim_imatge2)

        millor_malla_preliminar2 = guardar_millor_imatge_registrada(corregistre2,input, reference,
                                                                                 parametres_redimensionats,
                                                                                 path_carpeta_experiment, 10,
                                                                                 1/4, gamma,
                                                                                 None)


def guardar_millor_imatge_registrada(corregistre, imatge_input, imatge_reference, malla_vector,
                                     path_carpeta_experiment,
                                     iter, perturbacio, gamma ,diffstep):

    millors3resultats = corregistre.montecarlo(malla_vector, imatge_input, imatge_reference,
                                        path_carpeta_experiment,
                                        iter, perturbacio, gamma ,diffstep)
    valors_optims = millors3resultats[0]
    parametres_optims = millors3resultats[1]

    mx = corregistre.nx + 1
    my = corregistre.ny + 1

    i=0
    millor_malla_preliminar = []
    for parametres in parametres_optims:
        millor_malla_preliminar.append(corregistre.parametres_a_malla(parametres))
        imatge_registrada = corregistre.transformar(imatge_input, parametres)
        # visualitza_malla(imatge_registrada, millor_malla_preliminar[0], millor_malla_preliminar[1],
        #                 f'malla imatge registrada optima {mx},{my}',
        #                 f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
        imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{valors_optims[i]}.png',
            imatge_registrada)
        i = i+1
    return millor_malla_preliminar

if __name__ == '__main__':
    topti = time()
    main()
    tfi = time()
    print(tfi - topti)