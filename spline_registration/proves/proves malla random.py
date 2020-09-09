import numpy as np
from skimage.io import imread, imsave
from time import time
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab as pl
from skimage.filters import gaussian
from skimage.transform import resize
from spline_registration.transform_models import ElasticTransform
from spline_registration.utils import coordenades_originals,create_results_dir
from spline_registration.losses import RMSE,info_mutua
from scipy.optimize import least_squares



path_carpeta_experiment = create_results_dir('prova')


imatge_input = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

#imatge_input = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/seagull_input.jpg')
#imatge_reference = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/seagull_reference.jpg')

sigma= 10
imatge_input_gaussian = gaussian(imatge_input, sigma=sigma, multichannel=True)
imatge_reference_gaussian = gaussian(imatge_reference,sigma=sigma,multichannel=True )

path_imatge_input = f'{path_carpeta_experiment}/imatge_input.png'
path_imatge_reference = f'{path_carpeta_experiment}/imatge_reference.png'

imsave(path_imatge_input, imatge_input)
imsave(path_imatge_reference, imatge_reference)

corregistre = ElasticTransform()

import random

def funcio_min(parametres):
    global num_iteration

    malla_x = parametres[0: (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
    malla_y = parametres[(nx + 1) * (ny + 1): 2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))

    Coord_originals_x = coordenades_originals(imatge_input)[0]
    Coord_originals_y = coordenades_originals(imatge_input)[1]
    delta = [int((Coord_originals_x[-1] + 1) / nx) + 1, int((Coord_originals_y[-1] + 1) / ny) + 1]

    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y, nx, ny)

    imatge_registrada_input = corregistre.imatge_transformada(imatge_input, Coordenades_desti)
    # imatge_registrada_reference = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

    rmse = RMSE(imatge_registrada_input, imatge_reference)
    rmse_gaussian = RMSE(imatge_registrada_input, imatge_reference_gaussian)
    # rmse = RMSE (imatge_registrada_reference,imatge_input_gaussian)

    infomuta = info_mutua(imatge_registrada_input, imatge_reference_gaussian,10)

    mx_col_post = (malla_x[:, 1:])
    my_col_post = (malla_y[:, 1:])

    mx_fila_post = (malla_x[1:, :])
    my_fila_post = (malla_y[1:, :])

    d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
    d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1, :]), 2) + np.power((my_fila_post - malla_y[0:-1, :]), 2))

    distquadrat = np.sum(np.power(d1 - delta[1], 2)) + np.sum(np.power(d2 - delta[0], 2))
    distabs = np.sum(np.abs(d1 - delta[1])) + np.sum(np.abs(d2 - delta[0]))
    sd1 = np.std(d1)
    sd2 = np.std(d2)

    '''
    num_iteration = num_iteration + 1
    if num_iteration % 10 == 0:
        path_imatge_iteracio = f'{path_carpeta_experiment}/{num_iteration}.png'
        plt.imshow(imatge_registrada_input)
        pl.plot(malla_x, malla_y, color='green')
        pl.plot(malla_y, malla_x, color='green')
        pl.title('malla òptima sobre la imatge registrada')
        pl.savefig(path_imatge_iteracio)
        plt.close()
        #print(rmse,0.000003*distabs)
        print(rmse,0.001*(sd1+sd2))

    '''
    #print(rmse, 0.1*rmse_gaussian,0.001 * (sd1 + sd2))
    #return rmse + 0.1*rmse_gaussian + 0.001 * (sd1 + sd2)
    #rmse(referencia, input_corregistrada) + mu * rmse(referencia_gaussiana, input_corregistrada_gaussiana) +lambda *desviacions_estandard
    return -infomuta +(sd1+sd2) # + 0.001 * (sd1 + sd2)

def malla_monte_carlo(malla_x,malla_y, nx,ny,eps1, eps2, nombre_execucions):
    L = []
    malla_x_optima = malla_x
    malla_y_optima = malla_y
    min=20
    for l in range (0,nombre_execucions):
        epsilon1 = np.zeros(malla_x.shape)
        epsilon2 = np.zeros(malla_y.shape)

        for i in range(0, malla_x.shape[0]):
            for j in range(0, malla_x.shape[1]):
                epsilon1[i, j] = random.randrange(-eps1, eps1)

        for i in range(0, malla_y.shape[0]):
            for j in range(0, malla_y.shape[1]):
                epsilon2[i, j] = random.randrange(-eps2, eps2)

        malla_x_mod = malla_x + epsilon1
        malla_y_mod = malla_y + epsilon2

        x0 = np.concatenate((malla_x_mod.ravel(), malla_y_mod.ravel()))
        resultat = least_squares(funcio_min, x0,
                                        diff_step=0.002, method='trf', verbose=2)

        parametres_optims = resultat.x

        malla_x_mod = parametres_optims[0:(nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
        malla_y_mod = parametres_optims[(nx + 1) * (ny + 1):2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))

        Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x_mod, malla_y_mod, nx, ny)
        imatge_registrada_input = corregistre.imatge_transformada(imatge_input, Coordenades_desti)

        #rmse = RMSE(imatge_registrada_input, imatge_reference)
        rmse_gaussian = RMSE(imatge_registrada_input, imatge_reference_gaussian)
        infomuta = info_mutua(imatge_registrada_input, imatge_reference_gaussian, 10)

        path_imatge_registrada = f'{path_carpeta_experiment}/{nx}imatge_registrada{ infomuta }.png'
        imsave(path_imatge_registrada, imatge_registrada_input)


        #if rmse + 0.1*rmse_gaussian < min:
        if -infomuta < min:
            #min = rmse + 0.1*rmse_gaussian
            min = -infomuta
            malla_x_optima = malla_x_mod
            malla_y_optima = malla_y_mod



    return malla_x_optima,malla_y_optima

#AMPLIAR LA MALLA
def ampliacio_malla(malla_x,malla_y):
    dim = [2 * malla_x.shape[0] - 1, malla_x.shape[1]]
    malla_newx1 = np.zeros(dim)  # afegim les files entre dues de conegudes
    malla_newy1 = np.zeros(dim)

    filx = (malla_x[1:, :] + malla_x[0:-1, :]) / 2
    fily = (malla_y[1:, :] + malla_y[0:-1, :]) / 2

    for i in range(0, dim[0]):
        if (i % 2 == 0):
            malla_newx1[i] = malla_x[i // 2]
            malla_newy1[i] = malla_y[i // 2]

        else:
            malla_newx1[i] = filx[i // 2]
            malla_newy1[i] = fily[i // 2]

    colx = (malla_newx1[:, 1:] + malla_newx1[:, 0:-1]) / 2
    coly = (malla_newy1[:, 1:] + malla_newy1[:, 0:-1]) / 2

    dim = [2 * malla_x.shape[0] - 1, 2 * malla_x.shape[1] - 1]
    malla_newx2 = np.zeros(dim)  # afegim les columnes entre dues de conegudes
    malla_newy2 = np.zeros(dim)

    for i in range(0, dim[1]):
        if (i % 2 == 0):
            malla_newx2[:, i] = malla_newx1[:, i // 2]
            malla_newy2[:, i] = malla_newy1[:, i // 2]

        else:
            malla_newx2[:, i] = colx[:, i // 2]
            malla_newy2[:, i] = coly[:, i // 2]

    return malla_newx2, malla_newy2


nx = corregistre.nx
ny = corregistre.ny
mx = nx + 1
my = ny + 1

malla_vector = corregistre.malla_inicial(imatge_input)
malla_original = np.copy(malla_vector)

malla_x = malla_vector[0:(nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
malla_y = malla_vector[(nx + 1) * (ny + 1):2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))

Coord_originals_x = coordenades_originals(imatge_input)[0]
Coord_originals_y = coordenades_originals(imatge_input)[1]

delta = [int((Coord_originals_x[-1] + 1) / nx) + 1, int((Coord_originals_y[-1] + 1) / ny) + 1]
eps1 = int(delta[0] / 4)
eps2 =int(delta[1] / 4)

millor_malla_preliminar =  malla_monte_carlo(malla_x,malla_y, nx,ny,eps1, eps2,10)

malla_ampliada =  ampliacio_malla(millor_malla_preliminar[0],millor_malla_preliminar[1])

eps1 = int(delta[0] / 16)
eps2 =int(delta[1] / 16)
millor_malla_definitiva = malla_monte_carlo( malla_ampliada[0],malla_ampliada[1], 4,4,eps1,eps2,10)

Coordenades_desti2 = corregistre.posicio(Coord_originals_x, Coord_originals_y,millor_malla_definitiva[0], millor_malla_definitiva[1], 4, 4)

imatge_registrada_input= corregistre.imatge_transformada(imatge_input, Coordenades_desti2)
path_imatge_iteracio = f'{path_carpeta_experiment}/malla_imatge_registrada.png'
plt.imshow(imatge_registrada_input)
pl.plot(millor_malla_definitiva[0], millor_malla_definitiva[1], color='green')
pl.plot(millor_malla_definitiva[1],millor_malla_definitiva[0], color='green')
pl.title('malla òptima sobre la imatge registrada')
pl.savefig(path_imatge_iteracio)
plt.close()

path_imatge_registrada = f'{path_carpeta_experiment}/imatge_registrada.png'

imsave(path_imatge_registrada, imatge_registrada_input)
'''
L=[]
import random
for l in range(0,20):

    nx = corregistre.nx
    ny = corregistre.ny
    mx = nx + 1
    my = ny + 1

    malla_vector = corregistre.malla_inicial(imatge_input)
    malla_original = np.copy(malla_vector)

    malla_x = malla_vector[0:(nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
    malla_y = malla_vector[(nx + 1) * (ny + 1):2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))

    Coord_originals_x = coordenades_originals(imatge_input)[0]
    Coord_originals_y = coordenades_originals(imatge_input)[1]

    delta = [int((Coord_originals_x[-1] + 1) / nx) + 1, int((Coord_originals_y[-1] + 1) / ny) + 1]

    epsilon1 = np.zeros(malla_x.shape)
    epsilon2 = np.zeros(malla_y.shape)

    for i in range (0, malla_x.shape[0]):
        for j in range (0, malla_x.shape[1]):
            epsilon1[i,j] = random.randrange( -int(delta[0]/4), int(delta[0]/4))

    for i in range (0, malla_y.shape[0]):
        for j in range (0, malla_y.shape[1]):
            epsilon2[i,j] = random.randrange( -int(delta[0]/4), int(delta[0]/4))


    malla_x = malla_x + epsilon1
    malla_y = malla_y + epsilon2

    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y, nx, ny)

    #opcio1
    imatge_registrada_input = corregistre.imatge_transformada(imatge_input, Coordenades_desti)
    #opcio2
    #imatge_registrada_reference = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

    path_imatge_inicial = f'{path_carpeta_experiment}/{l}.png'
    plt.imshow(imatge_registrada_input)
    pl.plot(malla_x, malla_y, color='green')
    pl.plot(malla_y, malla_x, color='green')
    pl.title('malla inicial aleatòria')
    pl.savefig(path_imatge_inicial)
    plt.close()
    # print(rmse,0.000003*distabs)
    #print(rmse, 0.001 * (sd1 + sd2))


    #x0 = np.concatenate((malla_x_vec.ravel(), malla_y_vec.ravel()))
    x0 = np.concatenate((malla_x_vec, malla_y_vec))
    topti=time()
    resultat_opcio1 = least_squares(funcio_min, x0,
                                    diff_step=0.002, method='trf', verbose=2)

    #resultat_opcio1= least_squares(funcio_min, np.concatenate((malla_x_vec.ravel(), malla_y_vec.ravel())), diff_step=0.002, method='trf',verbose=2)
    tfi=time()
    print(resultat_opcio1, tfi-topti)

    parametres_optims = resultat_opcio1.x


    malla_x = parametres_optims[0:(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_y = parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_x_vec=parametres_optims[0:(nx+1)*(ny+1)]
    malla_y_vec=parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)]

    #epsilon1 = np.zeros(malla_x_vec.shape)
    #epsilon2 = np.zeros(malla_y_vec.shape)
    #for i in range(0, malla_x_vec.shape[0]):
    #    epsilon1[i] = random.randrange(-int(delta[0] / 8), int(delta[0] / 8))
    #for i in range(0, malla_y_vec.shape[0]):
     #   epsilon2[i] = random.randrange(-int(delta[1] / 8), int(delta[1] / 8))

    #malla_x_vec = malla_x_vec + epsilon1
    #malla_y_vec = malla_y_vec + epsilon2
    #malla_x = malla_x_vec.reshape((nx + 1), (ny + 1))
   # malla_y = malla_y_vec.reshape((nx + 1), (ny + 1))

    #Coord_originals_x = coordenades_originals(imatge_input)[0]
    #Coord_originals_y = coordenades_originals(imatge_input)[1]
    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y,nx,ny)


    imatge_registrada_input = corregistre.imatge_transformada(imatge_input, Coordenades_desti)
    #imatge_registrada_reference = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

    #plt.imshow(imatge_registrada_input)
    #pl.plot(malla_x,malla_y,color='green')
    #pl.plot(malla_y,malla_x,color = 'green')
    #pl.title('malla òptima sobre la imatge registrada')
    #plt.show()

    path_imatge_registrada = f'{path_carpeta_experiment}/imatge_registrada{l}.png'
    #path_imatge_diferencia = f'{path_carpeta_experiment}/imatge_diferencia.png'


    imsave(path_imatge_registrada,imatge_registrada_input)
    #imsave(path_imatge_registrada, imatge_registrada_reference)

    malla_new = np.mgrid[ 0: 2*delta[0] :int(delta[0]/2), 0: 2*delta[1]: int(delta[1]/2)]
    malla_new_vector = np.concatenate((malla_new[0].ravel(), malla_new[1].ravel()), axis=0)


    mx_col_post = (malla_x[:, 1:])
    my_col_post = (malla_y[:, 1:])
    mx_fila_post = (malla_x[1:, :])
    my_fila_post = (malla_y[1:, :])
    colx=(mx_col_post + malla_x[:,0:-1])/2
    coly=(my_col_post + malla_y[:,0:-1])/2

    filx=(mx_fila_post + malla_x[0:-1, :])/2
    fily=(my_fila_post + malla_y[0:-1, :])/2

    j1=0
    j2=0
    j3=0
    for i in [0,10,20]:
        for k in range (0,(2*mx-1),2):
            malla_new_vector[k + i]= malla_x_vec[j1]
            malla_new_vector[k + i + 25] = malla_y_vec[j1]
            j1=j1+1

    for i in [0, 10]:
        for k in range (1,(2*mx-1),2):

            malla_new_vector[k + i] = colx.flatten()[j2]
            malla_new_vector[k + i + 25] = coly.flatten()[j2]
            j2=j2+1

        for k in range ((2*mx-1),2*(2*mx-1),2):
            malla_new_vector[k + i] = filx.flatten()[j3]
            malla_new_vector[k + i + 25] = fily.flatten()[j3]

            j3=j3+1



    nx,ny= (4,4)
    resultat2iteracio = least_squares(funcio_min, malla_new_vector, diff_step=0.002, method='trf', verbose=2)
    parametres_optims2 = resultat2iteracio.x


    malla_x2 = parametres_optims2[0:(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_y2 = parametres_optims2[(nx+1)*(ny+1):2*(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_x_vec2=parametres_optims2[0:(nx+1)*(ny+1)]
    malla_y_vec2=parametres_optims2[(nx+1)*(ny+1):2*(nx+1)*(ny+1)]

    Coord_originals_x2 = coordenades_originals(imatge_input)[0]
    Coord_originals_y2 = coordenades_originals(imatge_input)[1]
    Coordenades_desti2 = corregistre.posicio(Coord_originals_x2, Coord_originals_y2, malla_x2, malla_y2,nx,ny)


    imatge_registrada_input2 = corregistre.imatge_transformada(imatge_input, Coordenades_desti2)
    #imatge_registrada_reference2 = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti2)

    #plt.imshow(imatge_registrada_input2)
    #pl.plot(malla_x2,malla_y2,color='green')
    #pl.plot(malla_y2,malla_x2,color = 'green')
    #pl.title('malla òptima sobre la imatge registrada')
    #plt.show()

    rmse = RMSE(imatge_registrada_input2, imatge_reference_gaussian)
    #rmse = RMSE(imatge_registrada_reference2, imatge_reference_gaussian)
    #infomutua = info_mutua(imatge_registrada_input, imatge_reference_gaussian, 10)
    L.append(rmse)
    #L.append((infomutua))

    path_imatge_iteracio = f'{path_carpeta_experiment}/({l}){rmse}.png'
    plt.imshow(imatge_registrada_input2 )
    pl.plot(malla_x2, malla_y2, color='green')
    pl.plot(malla_y2, malla_x2, color='green')
    pl.title('malla òptima sobre la imatge registrada')
    pl.savefig(path_imatge_iteracio)
    plt.close()

    path_imatge_registrada = f'{path_carpeta_experiment}/{rmse}imatge_registrada.png'
    #path_imatge_diferencia = f'{path_carpeta_experiment}/imatge_diferencia.png'

    imsave(path_imatge_registrada,imatge_registrada_input2 )
    
'''



'''
fitxer_sortida = open(f'{path_carpeta_experiment}/descripcio prova.txt', "w+")
fitxer_sortida.write(f'He començat amb una malla nx={nx}, ny ={ny} aleatòria (els punts originals es podem moure com a molt +-delta/4).\n'
                     f'la funció que minimitz: min_imatge_transformada'
                     f'amb error quatràtic més petit és:{min(L)} corresponent a la imatge: '
                     f'{path_carpeta_experiment}/{min(L)}imatge_registrada.png'
                     f'\n a naquest cas hem aplicat un filtratge gaussià a la  imatge_reference abans de res.'
                     f'\n el filtratge gaussià té sigma={sigma}.'
                     f'\n diff_step=0.002'
                     f'\n el resultat de la optimització: {resultat2iteracio}'
                     f'\n terme regularitzador, a partir de les distàncies entre valors consecutius a la malla '
                     f'el que faig és afegir la desviació (multiplicada per 0.001) entre les distàncies aixó tendeix a ser 0 quan hi ha menys desviació ')
fitxer_sortida.close()

'''


def visualize_side_by_side(self, image_top, image_bottom, path, title=None):
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.subplot(2, 1, 1)  # si les imatges són allargades millor
    plt.imshow(image_top)
    # plt.subplot(1, 2, 2)
    plt.subplot(2, 1, 2)  # si les imatges són allargades millor
    plt.imshow(image_bottom)
    if title:
        plt.title(title)
    plt.savefig(path)
    plt.close()


