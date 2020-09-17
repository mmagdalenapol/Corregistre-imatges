import numpy as np
from skimage.io import imread, imsave
from time import time
# from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab as pl
from skimage.filters import gaussian
from skimage.transform import rescale
import random
from spline_registration.transform_models import ElasticTransform_IM
from spline_registration.utils import coordenades_originals, create_results_dir, rescalar_imatge, color_a_grisos
from spline_registration.losses import RMSE, info_mutua
from scipy.optimize import least_squares, minimize
from PIL import Image

from spline_registration.utils import visualize_side_by_side


def main():
    # Inicialitzar dades

    for i in [1, 1998, 2020, 106, 201]:
        n = 2
        mida_malla1 = [n, n]
        pixels_per_vertex = 20

        # IMATGES ORIGINLAS
        imatge_input_orig = imread('seagull_input.jpg')
        imatge_ref_orig = imread('seagull_reference.jpg')
        dim_original = imatge_input_orig.shape

        # IMATGES RESCALADES amb les quals faig feina
        imatge_input = rescalar_imatge(imatge_input_orig, mida_malla1, pixels_per_vertex,True)
        imatge_reference = rescalar_imatge(imatge_ref_orig, mida_malla1, pixels_per_vertex,True)
        dim_imatge = imatge_input.shape
        # objecte de la classe ElasticTransform amb la dimensió de la malla inicial i de la imatge reduida.
        corregistre1 = ElasticTransform_IM(mida_malla1, dim_imatge)

        # IMATGES A ESCALA DE GRISOS
        imatge_input_gris = color_a_grisos(imatge_input)
        imatge_reference_gris = color_a_grisos(imatge_reference)

        # IMATGES GAUSSIANES
        imatge_input_gaussian = corregistre1.imatge_gaussian(imatge_input)
        imatge_reference_gaussian = corregistre1.imatge_gaussian(imatge_reference)

        imatge_input_gaussian_gris = corregistre1.imatge_gaussian(imatge_input_gris, False)
        imatge_reference_gaussian_gris = corregistre1.imatge_gaussian(imatge_reference_gris, False)

        # amb aquest for puc canviar fàcilment diferents valors com gamma o chi
        for chi in [0.08]:
            for gamma in [0.2]:
                perturbacio = 1 / 4
                input = imatge_input
                reference = imatge_reference

                # fix la llavor per tal d'obtenir sempre els mateixos resultats quan faig experimets amb els mateixos paràmetres
                random.seed(i)
                # Cream una carpeta per l'experiment i guardar les imatges
                path_carpeta_experiment = create_results_dir(f'{i}, gamma:{gamma},chi:{chi} ')
                imsave(f'{path_carpeta_experiment}/00_imatge_input.png', imatge_input)
                imsave(f'{path_carpeta_experiment}/00_imatge_reference.png', imatge_reference)
                imsave(f'{path_carpeta_experiment}/00_imatge_input_gris.png', imatge_input_gris)
                imsave(f'{path_carpeta_experiment}/00_imatge_reference_gris.png', imatge_reference_gris)
                imsave(f'{path_carpeta_experiment}/00_imatge_input_gaussian.png', imatge_input_gaussian)
                imsave(f'{path_carpeta_experiment}/00_imatge_reference_gaussian.png', imatge_reference_gaussian)
                imsave(f'{path_carpeta_experiment}/00_imatge_input_gaussian_gris.png', imatge_input_gaussian_gris)
                imsave(f'{path_carpeta_experiment}/00_imatge_reference_gaussian_gris.png',
                       imatge_reference_gaussian_gris)
                fitxer_sortida = open(f'{path_carpeta_experiment}/descripcio prova.txt', "w")

                # millor malla 3,3
                malla_vector1 = corregistre1.malla_inicial()
                millor_malla_preliminar1 = corregistre1.guardar_millor_imatge_registrada(input, reference,
                                                                                         malla_vector1,
                                                                                         path_carpeta_experiment,
                                                                                         fitxer_sortida, 5,
                                                                                         perturbacio, gamma, chi)

                # millor resultat malla (5,5)

                malla_ampliada1 = ampliacio_malla(millor_malla_preliminar1[0], millor_malla_preliminar1[1])
                malla_vector = [malla_ampliada1[0].ravel(), malla_ampliada1[1].ravel()]
                mida_malla2 = [2 * mida_malla1[0], 2 * mida_malla1[1]]

                input = rescalar_imatge(imatge_input_orig, mida_malla2, pixels_per_vertex - 10)
                reference =rescalar_imatge(imatge_ref_orig, mida_malla2, pixels_per_vertex - 10)
                dim_imatge2 = input.shape

                parametres_redimensionats = np.concatenate([dim_imatge2[0] / dim_imatge[0] * malla_vector[0],
                                                            dim_imatge2[1] / dim_imatge[1] * malla_vector[1]])
                corregistre2 = ElasticTransform_IM(mida_malla2, dim_imatge2)

                millor_malla_preliminar2 = corregistre2.guardar_millor_imatge_registrada(input, reference,
                                                                                         parametres_redimensionats,
                                                                                         path_carpeta_experiment,
                                                                                         fitxer_sortida, 2,
                                                                                         perturbacio, gamma, chi)

                # millor resultat malla (9,9)

                malla_ampliada2 = ampliacio_malla(millor_malla_preliminar2[0], millor_malla_preliminar2[1])
                malla_vector3 = [malla_ampliada2[0].ravel(), malla_ampliada2[1].ravel()]
                mida_malla3 = [2 * mida_malla2[0], 2 * mida_malla2[1]]

                input = rescalar_imatge(imatge_input_orig, mida_malla3, pixels_per_vertex - 15)
                reference = rescalar_imatge(imatge_ref_orig, mida_malla3, pixels_per_vertex - 15)
                dim_imatge3 = input.shape

                parametres_redimensionats2 = np.concatenate([dim_imatge3[0] / dim_imatge2[0] * malla_vector3[0],
                                                             dim_imatge3[1] / dim_imatge2[1] * malla_vector3[1]])
                corregistre3 = ElasticTransform_IM(mida_malla3, dim_imatge3)

                millor_malla_preliminar3 = corregistre3.guardar_millor_imatge_registrada(input, reference,
                                                                                         parametres_redimensionats2,
                                                                                         path_carpeta_experiment,
                                                                                         fitxer_sortida, 2, perturbacio,
                                                                                         gamma, chi)

                parametres_optims = [millor_malla_preliminar3[0].ravel(), millor_malla_preliminar3[1].ravel()]

                # passam a la escala de la imatge original el millor resultat
                parametres_redimensionats = np.concatenate([dim_original[0] / dim_imatge3[0] * parametres_optims[0],
                                                            dim_original[1] / dim_imatge3[1] * parametres_optims[1]])
                mx, my = corregistre3.nx + 1, corregistre3.ny + 1
                millor_malla = corregistre3.parametres_a_malla(parametres_redimensionats)
                corregistre4 = ElasticTransform_IM(mida_malla3, dim_original)
                imatge_registrada = corregistre4.transformar(imatge_input_orig, parametres_redimensionats)

                infomutua = np.sum(info_mutua(imatge_ref_orig,imatge_registrada,5))
                visualitza_malla(imatge_registrada, millor_malla[0], millor_malla[1],
                                 f'malla imatge registrada optima {mx, my}',
                                 f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
                imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{infomutua}.png',
                       imatge_registrada)

                im1 = Image.open(
                    '/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/seagull_reference.jpg').convert(
                    'L')
                im2 = Image.open(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{infomutua}.png').convert('L')
                im = Image.blend(im1, im2, 0.5)
                path_imatge_blend = f'{path_carpeta_experiment}/imatge_blend.png'
                im = im.save(path_imatge_blend)

                fitxer_sortida.write(
                    f'''
                Reduesc les imatges a imatges de resolució molt menor que depèn del nombre delements de la malla: {n}.\n
                pixels entre dos punts consecutius de la malla:{pixels_per_vertex}, factor perturbacio {perturbacio}\n
                A continuació guard les imatges a escala de grisos\n
                Calcul 100 imatges registrades a partir de 100 malles inicials aleatòries diferents de dimensió: {mida_malla1} +1 \n
                Ara a partir de la millor imatge corregistrada de les anteriors millor la malla inicial, ara de dimensió: mida_malla2 +1    
                i arrib a 10 imatges registrades\n 
                residus:residus imatges ,{gamma}*residuals_regularizacio
                calcul els residus a partir de les imatges a escala de color només (només els gaussians).\n
                la funció minimitzar té els següents paràmetres:   
                resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                            diff_step=None, gtol=1e-12, xtol=1e-13, ftol=1e-13, x_scale=1,
                                            method='lm', verbose=2)
                \n
                {millor_malla_preliminar1}\n
                {millor_malla_preliminar2}\n
                {millor_malla_preliminar3}
                ''')
                fitxer_sortida.close()




# AMPLIAR LA MALLA
def ampliacio_malla(malla_x, malla_y):
    filx = (malla_x[0:-1, :] + malla_x[1:, :]) / 2
    fily = (malla_y[0:-1, :] + malla_y[1:, :]) / 2

    dim = [2 * malla_x.shape[0] - 1, malla_x.shape[1]]
    fil_ampl_x = np.zeros(dim)  # afegim les files entre dues de conegudes
    fil_ampl_y = np.zeros(dim)

    for i in range(0, dim[0]):
        if (i % 2 == 0):  # als indexs parells deixam les files originals
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
        if (i % 2 == 0):  # als indexs parells deixam el conegut
            col_ampl_x[:, i] = fil_ampl_x[:, i // 2]
            col_ampl_y[:, i] = fil_ampl_y[:, i // 2]

        else:  # als indexs senars afegim els valors interpolats
            col_ampl_x[:, i] = colx[:, i // 2]
            col_ampl_y[:, i] = coly[:, i // 2]

    return col_ampl_x, col_ampl_y


def visualitza_malla(imatge, malla_x, malla_y, title,
                     path_guardar=None):  # podem fer tant la malla inicial com l'òptima
    plt.imshow(imatge)
    pl.plot(malla_x, malla_y, color='green')
    pl.plot(malla_y, malla_x, color='green')
    pl.title(title)
    if path_guardar:
        pl.savefig(path_guardar)
    plt.close()


# Posam això aquí abaix, que és el que farà que s'executi la funció main (i per tant, tota la "lògica")
if __name__ == '__main__':
    topti = time()
    main()
    tfi = time()
    print(tfi - topti)