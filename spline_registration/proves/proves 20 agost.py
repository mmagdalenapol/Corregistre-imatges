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
from spline_registration.utils import coordenades_originals,create_results_dir,parametres_a_malla
from spline_registration.losses import RMSE, info_mutua
from scipy.optimize import least_squares, minimize

from spline_registration.utils import visualize_side_by_side



def main():
    # Inicialitzar dades
    path_carpeta_experiment = create_results_dir('prova')
    imatge_input = imread('dog_input.jpg')
    imatge_reference = imread('dog_reference.jpg')

    corregistre = ElasticTransform()
    nx = corregistre.nx
    ny = corregistre.ny

    mx = nx + 1
    my = ny + 1

    # Reescalar imatges
    '''
    pixels_per_vertex = 25
    resolucio_ideal = (mx * pixels_per_vertex, my * pixels_per_vertex)
    scale_factor = (resolucio_ideal[0] / imatge_input.shape[0], resolucio_ideal[1] / imatge_input.shape[1])
    if scale_factor:
        imatge_input = rescale(imatge_input, scale_factor, multichannel=True,anti_aliasing=False)
        imatge_reference = rescale(imatge_reference, scale_factor,multichannel=True, anti_aliasing=False)    
    '''
    imatge_input = corregistre.rescalar_imatge(imatge_input)
    imatge_reference = corregistre.rescalar_imatge(imatge_reference)

    # Convertir imatges escala de grisos
    #imatge_input = color_a_grisos(imatge_input)
    #imatge_reference = color_a_grisos(imatge_reference)

    # Inicialitzar filtratge Gaussià
    '''
    sigma = (imatge_input.shape[0]/2 + imatge_input.shape[1]/2) * 1/5
    imatge_input_gaussian = gaussian(imatge_input, sigma=sigma, multichannel=True)
    imatge_reference_gaussian = gaussian(imatge_reference,sigma=sigma, multichannel=True)
    '''

    imatge_input_gaussian = corregistre.imatge_gaussian(imatge_input)
    imatge_reference_gaussian = corregistre.imatge_gaussian(imatge_reference)

    # Guardar algunes imatges
    imsave(f'{path_carpeta_experiment}/00_imatge_input.png', imatge_input)
    imsave(f'{path_carpeta_experiment}/00_imatge_reference.png', imatge_reference)
    imsave(f'{path_carpeta_experiment}/00_imatge_input_gaussian.png', imatge_input_gaussian)
    imsave(f'{path_carpeta_experiment}/00_imatge_reference_gaussian.png', imatge_reference_gaussian)

    malla_vector1 = corregistre.malla_inicial(imatge_input)

    parametres_optims1 = montecarlo(malla_vector1, imatge_input, imatge_reference, corregistre, nx, ny, path_carpeta_experiment,
               5)
    millor_malla_preliminar = parametres_a_malla(parametres_optims1,mx,my)

    malla_ampliada = ampliacio_malla(millor_malla_preliminar[0], millor_malla_preliminar[1])

    malla_vector2 = np.concatenate([malla_ampliada[0].ravel(), malla_ampliada[1].ravel()])

    parametres_optims2 = montecarlo(malla_vector2, imatge_input, imatge_reference, corregistre, 2*nx, 2*ny, path_carpeta_experiment,
               5)
    imatge_registrada_input = transformar(imatge_input, corregistre, 2*nx, 2*ny,parametres_optims2)

    fitxer_sortida = open(f'{path_carpeta_experiment}/descripcio prova.txt', "w+")
    fitxer_sortida.write(
        f'''
        Reduesc les imatges a imatges de resolució molt menor que depèn del nombre d'elements de la malla.\n
        A continuació guard les imatges amb un filtratge gaussià. La sigma emprada depèn de la dimensió de la imatge.\n
        Calcul 5 imatges registrades a partir de 5 malles inicials aleatòries diferents de dimensió (nx+1)*(ny+1)\n
        Ara a partir de la millor imatge corregistrada de les anteriors millor la malla inicial, ara de dimensió (2*nx+1)*(2*ny+1)  
        i arrib a 5 imatges registrades\n
        calcul els residus ponderats així: [residuals_rmse, residuals_gaussian_rmse, residuals_regularizacio]\n
        la funció minimitzar té els següents paràmetres:   
        resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                        diff_step=0.01, gtol=1e-8, xtol=1e-8, ftol=1e-8, x_scale=1,
                                        method='lm', verbose=2)
        ''' )
    fitxer_sortida.close()


def color_a_grisos(imatge):
    return 0.2125 * imatge[:, :, 0] + 0.7154 * imatge[:, :, 1] + 0.0721 * imatge[:, :, 2]
#Perquè aquests nombres? per internet he trobat aquests: 0.2989 * R + 0.5870 * G + 0.1140 *B

'''
def parametres_a_malla(parametres, mx, my):
    malla_x = parametres[0: mx * my].reshape(mx, my)
    malla_y = parametres[mx * my: 2 * mx * my].reshape(mx, my)
    return malla_x, malla_y

'''


def perturbar_malla_aleatoriament(malla_vector, mx, my, imatge_input, perturbacio=1/4):

    malla_x,malla_y = parametres_a_malla(malla_vector,mx,my)
    Coord_originals_x, Coord_originals_y = coordenades_originals(imatge_input)

    delta = [int((Coord_originals_x[-1] + 1) / (mx-1)) + 1, int((Coord_originals_y[-1] + 1) / (my-1)) + 1]

    epsilon1 = np.zeros(malla_x.shape)
    epsilon2 = np.zeros(malla_y.shape)
    if perturbacio:

        for i in range(0, malla_x.shape[0]):
            for j in range(0, malla_x.shape[1]):
                epsilon1[i,j] = random.randrange( -int(delta[0]*perturbacio), int(delta[0]*perturbacio))
        for i in range(0, malla_y.shape[0]):
            for j in range(0, malla_y.shape[1]):
                epsilon2[i,j] = random.randrange(-int(delta[1]*perturbacio), int(delta[1]*perturbacio))

    malla_x = malla_x + epsilon1
    malla_y = malla_y + epsilon2
    return malla_x, malla_y, Coord_originals_x, Coord_originals_y



#AMPLIAR LA MALLA
def ampliacio_malla(malla_x,malla_y):
    filx = (malla_x[0:-1, :] + malla_x[1:, :]) / 2
    fily = (malla_y[0:-1, :] + malla_y[1:, :]) / 2

    dim = [2 * malla_x.shape[0] - 1, malla_x.shape[1]]
    fil_ampl_x = np.zeros(dim)  # afegim les files entre dues de conegudes
    fil_ampl_y = np.zeros(dim)


    for i in range(0, dim[0]):
        if (i % 2 == 0): # als indexs parells deixam les files originals
            fil_ampl_x[i] = malla_x[i // 2]
            fil_ampl_y[i] = malla_y[i // 2]

        else: # als indexs senars afegim les files interpolades
            fil_ampl_x[i] = filx[i // 2]
            fil_ampl_y[i] = fily[i // 2]

    colx = (fil_ampl_x[:, 0:-1] + fil_ampl_x[:, 1:] ) / 2
    coly = (fil_ampl_y[:, 0:-1] + fil_ampl_y[:, 1:] ) / 2

    dim = [2 * malla_x.shape[0] - 1, 2 * malla_x.shape[1] - 1]
    col_ampl_x = np.zeros(dim)  # afegim les columnes entre dues de conegudes
    col_ampl_y = np.zeros(dim)

    for i in range(0, dim[1]):
        if (i % 2 == 0): #als indexs parells deixam el conegut
            col_ampl_x[:, i] = fil_ampl_x[:, i // 2]
            col_ampl_y[:, i] = fil_ampl_y[:, i // 2]

        else: #als indexs senars afegim els valors interpolats
            col_ampl_x[:, i] = colx[:, i // 2]
            col_ampl_y[:, i] = coly[:, i // 2]

    return col_ampl_x, col_ampl_y


def transformar(imatge, corregistre, nx, ny, parametres):
    malla_x, malla_y = parametres_a_malla(parametres, nx+1, ny+1)
    Coord_originals_x, Coord_originals_y = coordenades_originals(imatge)

    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y, nx, ny)

    return corregistre.imatge_transformada(imatge, Coordenades_desti)

num_iteration=0
def residus(parametres, imatge_input ,imatge_reference, corregistre, nx, ny):
    global num_iteration
    num_iteration += 1

    imatge_registrada = transformar(imatge_input,corregistre,nx,ny,parametres) #enviam les coord_originals de la imatge input a les coor_desti
    imatge_registrada_gaussian = corregistre.imatge_gaussian(imatge_registrada)
    imatge_reference_gaussian = corregistre.imatge_gaussian(imatge_reference)


    malla_x, malla_y = parametres_a_malla(parametres, nx+1, ny+1)
    mx_col_post = (malla_x[:, 1:])
    my_col_post = (malla_y[:, 1:])

    mx_fila_post = (malla_x[1:, :])
    my_fila_post = (malla_y[1:, :])


    d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
    d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1,:]), 2) + np.power((my_fila_post - malla_y[0:-1,:]), 2))
    sd1 = np.std(d1)
    sd2 = np.std(d2)

    residuals_rmse = (imatge_registrada - imatge_reference).flatten()
    residuals_gaussian_rmse = (imatge_registrada_gaussian - imatge_reference_gaussian).flatten()
    residuals_regularizacio = np.asarray([sd1, sd2])

    if num_iteration % 10 == 3:
        rmse = RMSE(imatge_registrada, imatge_input)
        rmse_gaussian = RMSE(imatge_registrada_gaussian, imatge_reference_gaussian)
        #visualize_side_by_side(image_left=imatge_registrada_input, image_right=imatge_input,
        #                       title=f'RMSE = {rmse}')
        #visualize_side_by_side(image_left=imatge_registrada_gaussian,
        #                       image_right=imatge_input_gaussian,
        #                       title=f'RMSE = {rmse_gaussian}')


    return np.concatenate([residuals_rmse, residuals_gaussian_rmse,10*residuals_regularizacio])
    # return error + 0.001*(sd1+sd2)


def visualitza_malla(imatge, malla_x, malla_y,title, path_guardar=None):#podem fer tant la malla inicial com l'òptima
    plt.imshow(imatge)
    pl.plot(malla_x, malla_y, color='green')
    pl.plot(malla_y, malla_x, color='green')
    pl.title(title)
    if path_guardar:
        pl.savefig(path_guardar)
    plt.close()

def montecarlo(malla_vector,imatge_input,imatge_reference,corregistre,nx,ny,path_carpeta_experiment,nombre_execucions):
    min = 20
    for num_exec in range(1, nombre_execucions):

        factor_perturbacio = 0 if num_exec == 1 else 1/num_exec
        malla_x, malla_y, Coord_originals_x, Coord_originals_y = perturbar_malla_aleatoriament(malla_vector,
                                                                                               nx+1, ny+1, imatge_input,
                                                                                               factor_perturbacio)

        # Visualitza corregistre inicial (malla amb perturbació aleatòria)
        imatge_registrada_input = transformar(imatge_input,corregistre,nx,ny,np.concatenate([malla_x.flatten(), malla_y.flatten()]))
        visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla inicial aleatòria',
                         path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_original{nx}')

        topti = time()
        funcio_min_residus = lambda x: residus(x, imatge_input,imatge_reference, corregistre, nx, ny)
        resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                        diff_step=None, gtol=1e-2, xtol=1e-13, ftol=1e-13, x_scale=1e4,
                                        method='lm', verbose=2)

        # funcio_min_escalar = lambda x: np.sum(residus(x, imatge_reference, imatge_input, corregistre, nx, ny, sigma)**2)
        # resultat_opcio1 = minimize(funcio_min_escalar, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
        #                            method='L-BFGS-B',
        #                            # options={'eps': 0.002, 'gtol': 1e-14}
        #                            )
        tfi = time()
        print(resultat, tfi-topti)

        # Visualitzar resultat
        parametres = resultat.x
        imatge_registrada_input = transformar(imatge_input, corregistre,nx, ny, parametres)
        rmse = RMSE(imatge_registrada_input, imatge_reference)

        imatge_registrada_gaussian = corregistre.imatge_gaussian(imatge_registrada_input)
        imatge_reference_gaussian = corregistre.imatge_gaussian(imatge_reference)
        rmse_gaussian = RMSE(imatge_registrada_gaussian,imatge_reference_gaussian)

        #print(f'''\nrmse:{rmse},10*rmse_gaussian:{10*rmse_gaussian},suma {rmse+10*rmse_gaussian}\n''')
        malla_x,malla_y = parametres_a_malla(parametres,nx+1,ny+1)
        visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla optima',
                         path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_optima{nx}')
        #visualize_side_by_side(image_left=imatge_registrada_input,
        #                       image_right=imatge_reference,
        #                       title=f'Ajust (t={tfi-topti} s')

        #imsave(path_imatge_registrada,imatge_registrada_input)
        imsave(f'{path_carpeta_experiment}/{num_exec:02d}_imatge_registrada{nx}_{rmse}.png', imatge_registrada_input)
        dif = np.abs(imatge_registrada_input- imatge_reference)
        imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel{nx}.png', dif)

        dif_gaussian = np.abs(corregistre.imatge_gaussian(imatge_reference) -
                              corregistre.imatge_gaussian(imatge_registrada_input))
        imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel_gaussian{nx}.png', dif_gaussian)

        if rmse < min:
            min = rmse
            parametres_optims = parametres
    return parametres_optims

# Posam això aquí abaix, que és el que farà que s'executi la funció main (i per tant, tota la "lògica")
if __name__ == '__main__':
    main()