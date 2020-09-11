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
from spline_registration.utils import coordenades_originals, create_results_dir,rescalar_imatge, color_a_grisos
from spline_registration.losses import RMSE, info_mutua
from scipy.optimize import least_squares, minimize

from spline_registration.utils import visualize_side_by_side



def main():
    # Inicialitzar dades

    for n in [2]:
    #pixels_per_vertex=30
        mida_malla1 = [n,n]
        pixels_per_vertex = 25

        #IMATGES ORIGINLAS
        imatge_input_orig = imread('dog_input.jpg')
        imatge_ref_orig = imread('dog_reference.jpg')
        dim_original = imatge_input_orig.shape

        #IMATGES RESCALADES amb les quals faig feina
        imatge_input = rescalar_imatge(imatge_input_orig,mida_malla1,pixels_per_vertex)
        imatge_reference = rescalar_imatge(imatge_ref_orig,mida_malla1,pixels_per_vertex)
        dim_imatge = imatge_input.shape
        # objecte de la classe ElasticTransform amb la dimensió de la malla inicial i la dimensió de la imatge reduida.
        corregistre1 = ElasticTransform(mida_malla1, dim_imatge)

        #IMATGES A ESCALA DE GRISOS
        imatge_input_gris = color_a_grisos(imatge_input)
        imatge_reference_gris = color_a_grisos(imatge_reference)

        #IMATGES GAUSSIANES
        imatge_input_gaussian = corregistre1.imatge_gaussian(imatge_input)
        imatge_reference_gaussian = corregistre1.imatge_gaussian(imatge_reference)

        imatge_input_gaussian_gris = corregistre1.imatge_gaussian(imatge_input_gris,False)
        imatge_reference_gaussian_gris = corregistre1.imatge_gaussian(imatge_reference_gris,False)

        #amb aquest for puc canviar fàcilment diferents valors com gamma o perturbacio
        for diffstep in [None]:
            gamma = 1
            perturbacio = 1/4
            input = imatge_input_gris
            reference = imatge_reference_gris
            input_orig = color_a_grisos(imatge_input_orig)
            reference_orig = color_a_grisos(imatge_ref_orig)
            multichannel = False

            #fix la llavor per tal d'obtenir sempre els mateixos resultats quan faig experimets amb els mateixos paràmetres
            random.seed(1)
            #Cream una carpeta per l'experiment i guardar les imatges
            path_carpeta_experiment = create_results_dir(f'{n},diff_step:{diffstep}')
            imsave(f'{path_carpeta_experiment}/00_imatge_input.png', imatge_input)
            imsave(f'{path_carpeta_experiment}/00_imatge_reference.png', imatge_reference)
            imsave(f'{path_carpeta_experiment}/00_imatge_input_gris.png', imatge_input_gris)
            imsave(f'{path_carpeta_experiment}/00_imatge_reference_gris.png', imatge_reference_gris)
            imsave(f'{path_carpeta_experiment}/00_imatge_input_gaussian.png', imatge_input_gaussian)
            imsave(f'{path_carpeta_experiment}/00_imatge_reference_gaussian.png', imatge_reference_gaussian)
            imsave(f'{path_carpeta_experiment}/00_imatge_input_gaussian_gris.png', imatge_input_gaussian_gris)
            imsave(f'{path_carpeta_experiment}/00_imatge_reference_gaussian_gris.png', imatge_reference_gaussian_gris)

            #millor malla 3,3
            malla_vector1 = corregistre1.malla_inicial()
            millor_malla_preliminar1 = corregistre1.guardar_millor_imatge_registrada( input, reference, malla_vector1,
                                                 path_carpeta_experiment,100, perturbacio, gamma,diffstep)





            #millor resultat malla (5,5)

            malla_ampliada1 = ampliacio_malla(millor_malla_preliminar1[0], millor_malla_preliminar1[1])
            malla_vector = [malla_ampliada1[0].ravel(), malla_ampliada1[1].ravel()]
            mida_malla2 = [2*mida_malla1[0], 2*mida_malla1[1]]
            input = rescalar_imatge(input_orig, mida_malla2, pixels_per_vertex-10, multichannel)
            reference = rescalar_imatge(reference_orig, mida_malla2, pixels_per_vertex-10, multichannel)

            dim_imatge2 = input.shape

            parametres_redimensionats = np.concatenate([dim_imatge2[0] / dim_imatge[0] * malla_vector[0],
                                                        dim_imatge2[1] / dim_imatge[1] * malla_vector[1]])
            corregistre2 = ElasticTransform(mida_malla2, dim_imatge2)

            millor_malla_preliminar2 = corregistre2.guardar_millor_imatge_registrada(input, reference,
                                                                                     parametres_redimensionats,
                                                                                     path_carpeta_experiment, 100,
                                                                                     perturbacio, gamma,
                                                                                     diffstep)


            #millor resultat malla (9,9)

            malla_ampliada2 = ampliacio_malla(millor_malla_preliminar2[0], millor_malla_preliminar2[1])
            malla_vector3 = [malla_ampliada2[0].ravel(), malla_ampliada2[1].ravel()]
            mida_malla3 = [2 * mida_malla2[0], 2 * mida_malla2[1]]

            input = rescalar_imatge(input_orig, mida_malla3, pixels_per_vertex-15,multichannel)
            reference = rescalar_imatge(reference_orig, mida_malla3, pixels_per_vertex-15,multichannel)
            dim_imatge3 = input.shape
            parametres_redimensionats2 = np.concatenate([dim_imatge3[0] / dim_imatge2[0] * malla_vector3[0] ,
                                                        dim_imatge3[1] / dim_imatge2[1] * malla_vector3[1]])
            corregistre3= ElasticTransform(mida_malla3, dim_imatge3)

            millor_malla_preliminar3 = corregistre3.guardar_millor_imatge_registrada(input, reference, parametres_redimensionats2,
                                                 path_carpeta_experiment,15, perturbacio, gamma,diffstep)




            parametres_optims=[millor_malla_preliminar3[0].ravel(),millor_malla_preliminar3[1].ravel()]

            #passam a la escala de la imatge original el millor resultat
            parametres_redimensionats = np.concatenate([dim_original[0] / dim_imatge3[0] * parametres_optims[0] ,
                                                        dim_original[1] / dim_imatge3[1] * parametres_optims[1]])
            mx , my = corregistre3.nx + 1 , corregistre3.ny + 1
            millor_malla = corregistre3.parametres_a_malla(parametres_redimensionats)
            corregistre4 = ElasticTransform(mida_malla3,dim_original)
            imatge_registrada = corregistre4.transformar(imatge_input_orig,parametres_redimensionats)

            rmse = RMSE(imatge_ref_orig,imatge_registrada)
            visualitza_malla(imatge_registrada, millor_malla[0], millor_malla[1],
                             f'malla imatge registrada optima {mx,my}',
                             f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
            imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx,my}_{rmse}.png',
                    imatge_registrada)

            fitxer_sortida = open(f'{path_carpeta_experiment}/descripcio prova.txt', "w+")
            fitxer_sortida.write(
             f'''
            Reduesc les imatges a imatges de resolució molt menor que depèn del nombre delements de la malla: {n}.\n
            pixels entre dos punts consecutius de la malla:{pixels_per_vertex}, factor perturbacio {perturbacio}\n
            A continuació guard les imatges a escala de grisos\n
            Calcul 100 imatges registrades a partir de 100 malles inicials aleatòries diferents de dimensió: {mida_malla1} +1 \n
            Ara a partir de la millor imatge corregistrada de les anteriors millor la malla inicial, ara de dimensió: mida_malla2 +1    
            i arrib a 20 imatges registrades\n 
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



'''
def perturbar_malla_aleatoriament(malla_vector, corregistre, imatge_input, perturbacio=1/4):

    malla_x,malla_y = corregistre.parametres_a_malla(malla_vector)
    Coord_originals_x, Coord_originals_y = coordenades_originals(imatge_input)

    delta = corregistre.delta

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


def transformar(imatge, corregistre,parametres):

    malla_x, malla_y = corregistre.parametres_a_malla(parametres)
    Coord_originals_x, Coord_originals_y = coordenades_originals(imatge)

    Coordenades_desti = corregistre.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)

    return corregistre.imatge_transformada(imatge, Coordenades_desti)
    num_iteration=0
def residus(parametres, imatge_input ,imatge_reference, corregistre,gamma):
    global num_iteration
    num_iteration += 1
    imatge_registrada = corregistre.transformar(imatge_input,parametres) #enviam les coord_originals de la imatge input a les coor_desti
    imatge_registrada_gaussian = corregistre.imatge_gaussian(imatge_registrada)
    imatge_reference_gaussian = corregistre.imatge_gaussian(imatge_reference)


    malla_x, malla_y = corregistre.parametres_a_malla(parametres)
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


    return np.concatenate([residuals_gaussian_rmse/sum(residuals_gaussian_rmse),gamma*residuals_regularizacio])
    #return np.concatenate([residuals_rmse/sum(residuals_rmse+residuals_gaussian_rmse),
                          # residuals_gaussian_rmse/sum(residuals_rmse+residuals_gaussian_rmse),
                           #gamma*residuals_regularizacio])
    # return error + 0.001*(sd1+sd2)
    
    
def montecarlo(malla_vector,imatge_input,imatge_reference,corregistre,path_carpeta_experiment,nombre_execucions,perturbacio,gamma):
    nx = corregistre.nx
    ny = corregistre.ny
    min = 20
    valors_optims = [0,0,0]
    parametres_optims = [0,0,0]
    execucio = -1

    for num_exec in range(1, nombre_execucions):

        execucio = execucio + 1
        #factor_perturbacio = 0 if num_exec == 1 else perturbacio
        factor_perturbacio = perturbacio
        malla_x, malla_y, Coord_originals_x, Coord_originals_y = corregistre.perturbar_malla_aleatoriament(malla_vector, imatge_input,
                                                                                               factor_perturbacio)



        # Visualitza corregistre inicial (malla amb perturbació aleatòria)
        
        #imatge_registrada_input = transformar(imatge_input,corregistre,np.concatenate([malla_x.flatten(), malla_y.flatten()]))

       3 visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla inicial aleatòria',
                         path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_original{nx}')
        
        




        topti = time()
        funcio_min_residus = lambda x: corregistre.residus(x, imatge_input,imatge_reference,gamma)
        resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                        diff_step=None, gtol=1e-12, xtol=1e-13, ftol=1e-13, x_scale=1,
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
        val_parametres = resultat.cost
        imatge_registrada_input = corregistre.transformar(imatge_input, parametres)
        rmse = RMSE(imatge_registrada_input, imatge_reference)

        imatge_registrada_gaussian = corregistre.imatge_gaussian(imatge_registrada_input)
        imatge_reference_gaussian = corregistre.imatge_gaussian(imatge_reference)
        rmse_gaussian = RMSE(imatge_registrada_gaussian,imatge_reference_gaussian)

        
        #malla_x,malla_y = parametres_a_malla(parametres,nx+1,ny+1)
        #visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla optima',
        #                 path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_optima{nx}')
        
        #visualize_side_by_side(image_left=imatge_registrada_input,
        #                       image_right=imatge_reference,
        #                       title=f'Ajust (t={tfi-topti} s')

        #imsave(path_imatge_registrada,imatge_registrada_input)
        imsave(f'{path_carpeta_experiment}/{num_exec:02d}_imatge_registrada{nx}_{val_parametres}.png', imatge_registrada_input)
        #dif = np.abs(imatge_registrada_input- imatge_reference)
        #imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel{nx}.png', dif)

        #dif_gaussian = np.abs(corregistre.imatge_gaussian(imatge_reference,False) -
       #                       corregistre.imatge_gaussian(imatge_registrada_input,False))
        #imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel_gaussian{nx}.png', dif_gaussian)

        #inicialitzar els valors i parametres_optims
        if execucio < 3:
            valors_optims[execucio] = val_parametres
            parametres_optims[execucio] = parametres

        if execucio == 2: #ordenar els valors i parametres associats.
            m1 = list(valors_optims)
            m2 = list(parametres_optims)

            valors_optims = sorted(valors_optims)
            for j in range (0,3):
                parametres_optims[j] = m2[m1.index(valors_optims[j])]

        if execucio > 2:
            if val_parametres < valors_optims[0]:
                valors_optims[1:] = valors_optims[0:-1]
                valors_optims[0] = val_parametres

                parametres_optims[1:]=parametres_optims[0:-1]
                parametres_optims[0] = parametres
        #if rmse_gaussian < min:
        #    min = rmse_gaussian
        #    parametres_optims = parametres

    return valors_optims,parametres_optims
    
def guardar_millor_imatge_registrada(corregistre,imatge_input, imatge_reference,malla_vector, path_carpeta_experiment,
                                     iter, perturbacio, gamma):
    millors3resultats = corregistre.montecarlo(malla_vector, imatge_input, imatge_reference, corregistre, path_carpeta_experiment,
                                   iter, perturbacio, gamma)
    valors_optims = millors3resultats[0]
    parametres_optims = millors3resultats[1][0]

    mx = corregistre.nx + 1
    my = corregistre.ny + 1
    millor_malla_preliminar = corregistre.parametres_a_malla(parametres_optims)

    imatge_registrada = corregistre.transformar(imatge_input,parametres_optims)
    visualitza_malla(imatge_registrada, millor_malla_preliminar[0], millor_malla_preliminar[1],
                     f'malla imatge registrada optima {mx},{my}',
                     f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
    rmse = RMSE(imatge_reference, imatge_registrada)
    imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{rmse}.png',
           imatge_registrada)
    return millor_malla_preliminar

'''



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


def resultat_malla_ampliada(malla_preliminar, dim_imatge_previa, input_orig, reference_orig, pixels_per_vertex,
                            path_carpeta_experiment, exec,perturbacio, gamma, diffstep, multichannel):
    malla_ampliada = ampliacio_malla(malla_preliminar[0], malla_preliminar[1])
    mida_malla_preliminar = [malla_ampliada.shape[0] - 1, malla_ampliada.shape[1] - 1]

    malla_vector = [malla_ampliada[0].ravel(), malla_ampliada[1].ravel()]
    mida_malla = [2 * mida_malla_preliminar[0], 2 * mida_malla_preliminar[1]]
    input = rescalar_imatge(input_orig, mida_malla, pixels_per_vertex, multichannel)
    reference = rescalar_imatge(reference_orig, mida_malla, pixels_per_vertex, multichannel)

    dim_imatge2 = input.shape

    parametres_redimensionats = np.concatenate([dim_imatge2[0] / dim_imatge_previa[0] * malla_vector[0],
                                                dim_imatge2[1] / dim_imatge_previa[1] * malla_vector[1]])
    corregistre2 = ElasticTransform(mida_malla, dim_imatge2)

    millor_malla_preliminar2 = corregistre2.guardar_millor_imatge_registrada(input, reference,
                                                                             parametres_redimensionats,
                                                                             path_carpeta_experiment, exec,
                                                                             perturbacio, gamma,
                                                                             diffstep)
    return millor_malla_preliminar2

def visualitza_malla(imatge, malla_x, malla_y,title, path_guardar=None):#podem fer tant la malla inicial com l'òptima
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