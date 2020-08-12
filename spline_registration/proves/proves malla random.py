import numpy as np
from skimage.io import imread, imsave
from time import time
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab as pl
from skimage.filters import gaussian
from spline_registration.transform_models import ElasticTransform
from spline_registration.utils import coordenades_originals
from spline_registration.utils import create_results_dir
from spline_registration.losses import RMSE
from scipy.optimize import least_squares

path_carpeta_experiment = create_results_dir('prova')


imatge_input = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_reference = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')

sigma= 10
imatge_input_gaussian = gaussian(imatge_input, sigma=sigma, multichannel=True)
imatge_reference_gaussian = gaussian(imatge_reference,sigma=sigma,multichannel=True )

path_imatge_input = f'{path_carpeta_experiment}/imatge_input.png'
path_imatge_reference = f'{path_carpeta_experiment}/imatge_reference.png'

imsave(path_imatge_input, imatge_input)
imsave(path_imatge_reference, imatge_reference)

corregistre = ElasticTransform()


L=[]
import random
for l in range(0,15):

    nx = corregistre.nx
    ny = corregistre.ny
    mx = nx + 1
    my = ny + 1

    malla_vector = corregistre.malla_inicial(imatge_input)
    malla_original = np.copy(malla_vector)

    malla_x = malla_vector[0:(nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
    malla_y = malla_vector[(nx + 1) * (ny + 1):2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
    malla_x_vec = malla_x.flatten()
    malla_y_vec = malla_y.flatten()

    Coord_originals_x = coordenades_originals(imatge_input)[0]
    Coord_originals_y = coordenades_originals(imatge_input)[1]

    delta = [int((Coord_originals_x[-1] + 1) / nx) + 1, int((Coord_originals_y[-1] + 1) / ny) + 1]

    coord1=np.zeros(malla_x_vec.shape)
    coord2= np.zeros(malla_y_vec.shape)
    for i in range(0,malla_x_vec.shape[0]):
        coord1[i]=random.randrange(-int(delta[0]/4),int(delta[0]/4))
    for i in range (0,malla_y_vec.shape[0]):
        coord2[i]=random.randrange(-int(delta[1]/4),int(delta[1]/4))

    malla_x_vec = malla_x_vec + coord1
    malla_y_vec = malla_y_vec + coord2
    malla_x = malla_x_vec .reshape((nx + 1), (ny + 1))
    malla_y = malla_y_vec.reshape((nx + 1), (ny + 1))

    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y,malla_x, malla_y,nx,ny)

    #opcio1
    imatge_registrada_input = corregistre.imatge_transformada(imatge_input, Coordenades_desti)

    path_imatge_iteracio = f'{path_carpeta_experiment}/{l}.png'
    plt.imshow(imatge_registrada_input)
    pl.plot(malla_x, malla_y, color='green')
    pl.plot(malla_y, malla_x, color='green')
    pl.title('malla òptima sobre la imatge registrada')
    pl.savefig(path_imatge_iteracio)
    plt.close()
    # print(rmse,0.000003*distabs)
    #print(rmse, 0.001 * (sd1 + sd2))




    #num_iteration=0
    def funcio_min(parametres):
        global num_iteration

        malla_x = parametres[0:(nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))
        malla_y = parametres[(nx + 1) * (ny + 1): 2 * (nx + 1) * (ny + 1)].reshape((nx + 1), (ny + 1))

        Coord_originals_x = coordenades_originals(imatge_input)[0]
        Coord_originals_y = coordenades_originals(imatge_input)[1]
        delta = [int((Coord_originals_x [-1]+1) / nx) + 1, int((Coord_originals_y [-1] + 1) / ny) + 1]

        Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y,nx,ny )

        imatge_registrada_input= corregistre.imatge_transformada(imatge_input, Coordenades_desti)
    #  imatge_registrada_reference = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

        rmse = RMSE(imatge_registrada_input, imatge_reference_gaussian)
        #rmse = RMSE (imatge_registrada_reference,imatge_input_gaussian)

        n = min(nx, ny)

        mx_col_post = (malla_x[:, 1:])[0:n, 0:n]
        my_col_post = (malla_y[:, 1:])[0:n, 0:n]

        mx_fila_post = (malla_x[1:, :])[0:n, 0:n]
        my_fila_post = (malla_y[1:, :])[0:n, 0:n]


        d1 = np.sqrt(np.power((mx_col_post - malla_x[0:n, 0:n]), 2) + np.power((my_col_post - malla_y[0:n, 0:n]), 2))
        d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:n, 0:n]), 2) + np.power((my_fila_post - malla_y[0:n, 0:n]), 2))

        distquadrat = np.sum(np.power(d1-delta[1],2)) + np.sum(np.power(d2-delta[0],2))
        distabs = np.sum(np.abs(d1-delta[1]))+np.sum(np.abs(d2-delta[0]))
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


        #print(rmse, 0.169*(sd1+sd2))

        #return rmse +0.000003*distabs
        return rmse +0.001*(sd1+sd2)
        #+ 0.169*(sd1+sd2)


    topti=time()
    resultat_opcio1= least_squares(funcio_min, np.concatenate((malla_x_vec.ravel(), malla_y_vec.ravel())), diff_step=0.002, method='trf',verbose=2)
    tfi=time()
    print(resultat_opcio1, tfi-topti)

    parametres_optims = resultat_opcio1.x


    malla_x = parametres_optims[0:(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_y = parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)].reshape((nx+1),(ny+1))
    malla_x_vec=parametres_optims[0:(nx+1)*(ny+1)]
    malla_y_vec=parametres_optims[(nx+1)*(ny+1):2*(nx+1)*(ny+1)]

    coord1 = np.zeros(malla_x_vec.shape)
    coord2 = np.zeros(malla_y_vec.shape)
    for i in range(0, malla_x_vec.shape[0]):
        coord1[i] = random.randrange(-int(delta[0] / 8), int(delta[0] / 8))
    for i in range(0, malla_y_vec.shape[0]):
        coord2[i] = random.randrange(-int(delta[1] / 8), int(delta[1] / 8))

    malla_x_vec = malla_x_vec + coord1
    malla_y_vec = malla_y_vec + coord2
    malla_x = malla_x_vec.reshape((nx + 1), (ny + 1))
    malla_y = malla_y_vec.reshape((nx + 1), (ny + 1))

    Coord_originals_x = coordenades_originals(imatge_input)[0]
    Coord_originals_y = coordenades_originals(imatge_input)[1]
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

    imsave(path_imatge_input,imatge_input)
    imsave(path_imatge_reference,imatge_reference)
    imsave(path_imatge_registrada,imatge_registrada_input)


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
    #imatge_registrada_reference = corregistre.colors_transform_nearest_neighbours(imatge_reference, Coordenades_desti)

    #plt.imshow(imatge_registrada_input2)
    #pl.plot(malla_x2,malla_y2,color='green')
    #pl.plot(malla_y2,malla_x2,color = 'green')
    #pl.title('malla òptima sobre la imatge registrada')
    #plt.show()

    rmse = RMSE(imatge_registrada_input, imatge_reference_gaussian)
    L.append(rmse)

    path_imatge_iteracio = f'{path_carpeta_experiment}/({l}){rmse}.png'
    plt.imshow(imatge_registrada_input2)
    pl.plot(malla_x2, malla_y2, color='green')
    pl.plot(malla_y2, malla_x2, color='green')
    pl.title('malla òptima sobre la imatge registrada')
    pl.savefig(path_imatge_iteracio)
    plt.close()

    path_imatge_registrada = f'{path_carpeta_experiment}/{rmse}imatge_registrada.png'
    #path_imatge_diferencia = f'{path_carpeta_experiment}/imatge_diferencia.png'

    imsave(path_imatge_registrada,imatge_registrada_input2)

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
