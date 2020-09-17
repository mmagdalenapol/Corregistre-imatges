from skimage.transform import resize, rescale
from skimage.filters import gaussian
from skimage.io import imread, imsave
from spline_registration.utils import coordenades_originals, imatge_vec,color_a_grisos
import numpy as np
import random
from scipy.optimize import least_squares
from spline_registration.losses import RMSE, info_mutua
from skimage import feature


class BaseTransform:

    def find_best_transform(self, reference_image, input_image):
        raise NotImplementedError

    def apply_transform(self,reference_image, input_image):
        raise NotImplementedError

    def visualize_transform(self):
        return None

'''
class Rescala(BaseTransform):
    def __init__(self):
        self.dim_imatge = None

    def find_best_transform(self, reference_image, input_image):
        #self.dim_imatge = reference_image.shape

    def apply_transform(self, input_image):
        return resize(input_image,self.dim_imatge)


transformada = Rescala()

BaseTransform.apply_transform(transformada, input_image=None)
transformada.apply_transform(None)

'''

class Rescala(BaseTransform):

    def __init__(self):
        self.dim_image = None

    def find_best_transform(self, reference_image, input_image):
        return reference_image.shape

    def apply_transform(self, reference_image,input_image):
        return resize(input_image, reference_image.shape)




class ElasticTransform(BaseTransform):
    def __init__(self, mida_malla,  dim_imatge):
        self.dim_imatge = dim_imatge
        self.A = None
        self.nx = mida_malla[0]
        self.ny = mida_malla[1]
        self.delta = [int(dim_imatge[0]/mida_malla[0]) + 1, int(dim_imatge[1]/mida_malla[1]) + 1]

    '''
        def rescalar_imatge(self,imatge):
        resolucio_ideal = ((self.nx) * self.pixels_per_vertex, (self.ny) * self.pixels_per_vertex)
        scale_factor = (resolucio_ideal[0] / imatge.shape[0], resolucio_ideal[1] / imatge.shape[1])
        imatge_rescalada = rescale(imatge, scale_factor, multichannel=True,anti_aliasing=False)
        return imatge_rescalada

    '''


    def imatge_gaussian(self,imatge,multichanel=True):
        sigma = (imatge.shape[0] / 3 + imatge.shape[1] / 3) * 1 / 5
        imatge_gaussian = gaussian(imatge, sigma=sigma, multichannel=multichanel)
        return imatge_gaussian

    def edges(self,imatge):
        sigma = (imatge.shape[0] / 10 + imatge.shape[1] / 10) * 1 / 5
        imatge_gaussian = gaussian(imatge,sigma = sigma, multichannel = False)
        edges = feature.canny (imatge_gaussian)
        edges = np.where(edges==True,1,0)
        return edges

    def parametres_a_malla(self,parametres):
        files = self.nx + 1
        columnes = self.ny + 1
        malla_x = parametres[0: files * columnes].reshape(files, columnes)
        malla_y = parametres[files * columnes: 2 * files * columnes].reshape(files, columnes)
        return malla_x, malla_y

    def malla_inicial(self):
        nx = self.nx
        ny = self.ny
        delta = self.delta

        '''
        el +1 ens permet assegurar que la darrera fila/columna de la malla estan defora de la imatge.
        Ja que així creant aquests punts ficticis a fora podem interpolar totes les posicions de la imatge. 
        Ara la malla serà (nx+1)*(ny+1) però la darrera fila i la darrera columna com he dit són per tècniques.
        '''
        malla = np.mgrid[ 0: (nx+1)*delta[0] :delta[0], 0: (ny+1)*delta[1]:delta[1]]
        malla_x = malla[0]  # inicialitzam a on van les coordenades x a la imatge_reference
        malla_y = malla[1]  # inicialitzam a on van les coordenades y a la imatge_reference
        malla_vector = np.concatenate((malla_x.ravel(), malla_y.ravel()), axis=0)

        return malla_vector

    def perturbar_malla_aleatoriament(self, malla_vector,imatge_input, perturbacio=1 / 4):

        malla_x, malla_y = self.parametres_a_malla(malla_vector)
        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge_input)

        delta = self.delta

        epsilon1 = np.zeros(malla_x.shape)
        epsilon2 = np.zeros(malla_y.shape)
        if perturbacio:

            for i in range(0, malla_x.shape[0]):
                for j in range(0, malla_x.shape[1]):
                    epsilon1[i, j] = random.randrange(-int(delta[0] * perturbacio), int(delta[0] * perturbacio))
            for i in range(0, malla_y.shape[0]):
                for j in range(0, malla_y.shape[1]):
                    epsilon2[i, j] = random.randrange(-int(delta[1] * perturbacio), int(delta[1] * perturbacio))

        malla_x = malla_x + epsilon1
        malla_y = malla_y + epsilon2
        return malla_x, malla_y, Coord_originals_x, Coord_originals_y

    def posicio(self, x, y, malla_x, malla_y):
        # s val 0 quan la x està a coordenadesx
        # t val 0 quan la y està a coordenadesy
        # i index de la posició més pròxima per davall de la coordenada x a la malla
        # j index de la posició més pròxima per davall de la coordenada y a la malla
        #nx = self.nx
        #ny = self.ny
        '''
        hem d'interpolar totes les posicions de la imatge per tant imatge_input.shape[0] = x[-1]+1 
        i imatge_input.shape[1] = y[-1] + 1.
        '''
        nx = self.nx
        ny = self.ny
        delta = self.delta

        s, i = np.modf(x / delta[0])
        t, j = np.modf(y / delta[1])
        i = np.minimum(np.maximum(i.astype('int'), 0), nx)
        j = np.minimum(np.maximum(j.astype('int'), 0), ny)

        interpolacio = np.array([(s - 1) * (t - 1) * malla_x[i, j] + s * (1 - t) * malla_x[i + 1, j]
                                 + (1 - s) * t * malla_x[i, j + 1] + s * t * malla_x[i + 1, j + 1],
                                 (s - 1) * (t - 1) * malla_y[i, j] + s * (1 - t) * malla_y[i + 1, j]
                                 + (1 - s) * t * malla_y[i, j + 1] + s * t * malla_y[i + 1, j + 1]
                                 ])

        return interpolacio


    def imatge_transformada(self,imatge, coord_desti):
        '''
        Introduim la imatge_input i les coordenades a les quals es mouen les originals després d'aplicar l'interpolació.
        El que volem es tornar la imatge registrada que tengui a les coordenades indicades els colors originals:

        Per fer-ho definesc una imatge registrada (inicialment tota negre) i a les coordenades del destí
        anar enviant els colors originals.
        '''

        coord_desti = np.round(coord_desti).astype('int')  # Discretitzar
        coord_desti = np.maximum(coord_desti, 0)
        coord_desti[0] = np.minimum(coord_desti[0], imatge.shape[0] - 1)
        coord_desti[1] = np.minimum(coord_desti[1], imatge.shape[1] - 1)

        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge)

        registered_image = np.zeros_like(imatge)
        registered_image[Coord_originals_x, Coord_originals_y] = imatge[coord_desti[0], coord_desti[1]]
        return registered_image

    def transformar(self,imatge, parametres):

        malla_x, malla_y = self.parametres_a_malla(parametres)
        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge)

        Coordenades_desti = self.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y)

        return self.imatge_transformada(imatge, Coordenades_desti)

    def residus(self,parametres, imatge_input, imatge_reference, gamma, chi):

        imatge_registrada = self.transformar(imatge_input,parametres)  # enviam les coord_originals de la imatge input a les coor_desti
        imatge_registrada_gaussian = self.imatge_gaussian(imatge_registrada)
        imatge_reference_gaussian = self.imatge_gaussian(imatge_reference)


        edges_registrada = self.edges(imatge_registrada)
        edges_reference = self.edges(imatge_reference)
        sum_edges_registrada = np.sum(edges_registrada)
        sum_edges_reference = np.sum(edges_reference)
        dif_edge= np.abs(edges_registrada-edges_reference)


        malla_x, malla_y = self.parametres_a_malla(parametres)

        mx_col_post, my_col_post = malla_x[:, 1:], malla_y[:, 1:]
        mx_fila_post, my_fila_post = malla_x[1:, :], malla_y[1:, :]

        d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
        d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1, :]), 2) + np.power((my_fila_post - malla_y[0:-1, :]), 2))
        sd1 = np.std(d1)
        sd2 = np.std(d2)

        residuals_rmse = np.power((imatge_registrada - imatge_reference).flatten(),2)
        residuals_gaussian_rmse = np.power((imatge_registrada_gaussian - imatge_reference_gaussian).flatten(),2)
        residuals_regularizacio = np.asarray([sd1, sd2])
        residuals_edge = np.sum(dif_edge)


        #return np.concatenate([residuals_gaussian_rmse / sum(residuals_gaussian_rmse), gamma * residuals_regularizacio])
        den = sum(residuals_rmse+residuals_gaussian_rmse)
        if den == 0:
            den=1
        beta = 1-gamma-chi
        return np.concatenate([beta*residuals_rmse/den,
         beta*residuals_gaussian_rmse/den,
         gamma*residuals_regularizacio,
         [chi *residuals_edge/(sum_edges_reference+ (sum_edges_registrada - sum_edges_reference))] ])


    def colors_transform_nearest_neighbours(self,imatge_reference, Coordenades_desti):
            Coordenades_desti = np.round( Coordenades_desti).astype('int')  # Discretitzar
            Coordenades_desti = np.maximum(Coordenades_desti, 0)
            Coordenades_desti[0] = np.minimum(Coordenades_desti[0], imatge_reference.shape[0] - 1)
            Coordenades_desti[1] = np.minimum(Coordenades_desti[1], imatge_reference.shape[1] - 1)

            registered_image = imatge_reference[Coordenades_desti[0], Coordenades_desti[1]]
            registered_image = registered_image.reshape(imatge_reference.shape, order='F')

            return registered_image

    def montecarlo(self, malla_vector, imatge_input, imatge_reference, path_carpeta_experiment,fitxer_sortida,
                   nombre_execucions, perturbacio, gamma, chi):
        nx = self.nx
        ny = self.ny


        #enlloc de quedarme tan sols amb el valor mínim i els seus corresponents paràmetre vull que em guardi
        #la informació respectiva als tres valors més petits. Per si així arrib a millors resultats.
        valors_optims = [0, 0, 0]
        parametres_optims = [0, 0, 0]
        execucio = -1

        for num_exec in range(1, nombre_execucions):

            execucio = execucio + 1
            factor_perturbacio = 0 if num_exec == 1 else perturbacio
            #factor_perturbacio = perturbacio
            malla_x, malla_y, Coord_originals_x, Coord_originals_y = self.perturbar_malla_aleatoriament(
                malla_vector, imatge_input,
                factor_perturbacio)

            # Visualitza corregistre inicial (malla amb perturbació aleatòria)
            '''
            imatge_registrada_input = transformar(imatge_input,corregistre,np.concatenate([malla_x.flatten(), malla_y.flatten()]))

            visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla inicial aleatòria',
                             path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_original{nx}')

            '''
            funcio_min_residus = lambda x: self.residus(x, imatge_input, imatge_reference, gamma, chi)
            resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                     diff_step = None, gtol=1e-12, xtol=1e-13, ftol=1e-13,
                                     method='lm', verbose=2)

            # funcio_min_escalar = lambda x: np.sum(residus(x, imatge_reference, imatge_input, corregistre, nx, ny, sigma)**2)
            # resultat_opcio1 = minimize(funcio_min_escalar, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
            #                            method='L-BFGS-B',
            #                            # options={'eps': 0.002, 'gtol': 1e-14}
            #                            )

            # Visualitzar resultat
            parametres = resultat.x
            val_parametres = resultat.cost
            residus = funcio_min_residus(parametres)
            m = self.dim_imatge[0] * self.dim_imatge[1]

            residuals_rmse = residus[0:2*m]
            sd_malla = residus[-3:-1]
            contorn = residus[-1]


            fitxer_sortida.write(f'''
            \n{gamma,chi}:{num_exec}:
            residus rmse conjunts normalitzats:{residuals_rmse[34]},
            regularitzacio malla : {sd_malla},
            contorn : {contorn}\n
            ''')
            imatge_registrada_input = self.transformar(imatge_input, parametres)
            #rmse = RMSE(imatge_registrada_input, imatge_reference)

            imatge_registrada_gaussian = self.imatge_gaussian(imatge_registrada_input)
            imatge_reference_gaussian = self.imatge_gaussian(imatge_reference)
            #rmse_gaussian = RMSE(imatge_registrada_gaussian, imatge_reference_gaussian)

            '''
            malla_x,malla_y = parametres_a_malla(parametres,nx+1,ny+1)
            visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla optima',
                             path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_optima{nx}')
            '''


            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_imatge_registrada{nx}_{val_parametres}.png',
                   imatge_registrada_input)
            # dif = np.abs(imatge_registrada_input- imatge_reference)
            # imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel{nx}.png', dif)

            # dif_gaussian = np.abs(corregistre.imatge_gaussian(imatge_reference,False) -
            #                       corregistre.imatge_gaussian(imatge_registrada_input,False))
            # imsave(f'{path_carpeta_experiment}/{num_exec:02d}_error_per_pixel_gaussian{nx}.png', dif_gaussian)

            # inicialitzar els valors i parametres_optims

            #edges

            edges_registrada = self.edges(imatge_registrada_input)
            edges_reference = self.edges(imatge_reference)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_contorn_registrada_{nx}.png', edges_registrada)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_contorn_reference_{nx}.png', edges_reference)

            if execucio < 3:
                valors_optims[execucio] = val_parametres
                parametres_optims[execucio] = parametres

            if execucio == 2:  # ordenar els valors i parametres associats.
                m1 = list(valors_optims)
                m2 = list(parametres_optims)

                valors_optims = sorted(valors_optims)
                for j in range(0, 3):
                    parametres_optims[j] = m2[m1.index(valors_optims[j])]

            if execucio > 2:
                if val_parametres < valors_optims[0]:
                    valors_optims[1:] = valors_optims[0:-1]
                    valors_optims[0] = val_parametres

                    parametres_optims[1:] = parametres_optims[0:-1]
                    parametres_optims[0] = parametres
            # if rmse_gaussian < min:
            #    min = rmse_gaussian
            #    parametres_optims = parametres

        return valors_optims, parametres_optims

    def guardar_millor_imatge_registrada(self, imatge_input, imatge_reference, malla_vector,
                                         path_carpeta_experiment,fitxer_sortida,
                                         iter, perturbacio, gamma, chi):
        millors3resultats = self.montecarlo(malla_vector, imatge_input, imatge_reference,
                                            path_carpeta_experiment,fitxer_sortida,
                                            iter, perturbacio, gamma, chi)
        valors_optims = millors3resultats[0]
        parametres_optims = millors3resultats[1][0]

        mx = self.nx + 1
        my = self.ny + 1
        millor_malla_preliminar = self.parametres_a_malla(parametres_optims)

        imatge_registrada = self.transformar(imatge_input, parametres_optims)
        #visualitza_malla(imatge_registrada, millor_malla_preliminar[0], millor_malla_preliminar[1],
        #                 f'malla imatge registrada optima {mx},{my}',
        #                 f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
        #rmse = RMSE(imatge_reference, imatge_registrada)
        imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{valors_optims[0]}.png',
               imatge_registrada)
        return millor_malla_preliminar

    def find_best_transform(self, reference_image, input_image):

        return None

    def apply_transform(self,reference_image,input_image ):

        return None

class ElasticTransform_IM (ElasticTransform):

    def edges(self,imatge):
        sigma = (imatge.shape[0] / 10 + imatge.shape[1] / 10) * 1 / 5
        imatge_gaussian = gaussian(color_a_grisos(imatge),sigma = sigma, multichannel = False)
        edges = feature.canny (imatge_gaussian)
        edges = np.where(edges==True,1,0)
        return edges

    def residus(self,parametres, imatge_input, imatge_reference, gamma, chi):

        imatge_registrada = self.transformar(imatge_input,parametres)  # enviam les coord_originals de la imatge input a les coor_desti
        #imatge_registrada_gaussian = self.imatge_gaussian(imatge_registrada)
        #imatge_reference_gaussian = self.imatge_gaussian(imatge_reference)


        edges_registrada = self.edges(imatge_registrada)
        edges_reference = self.edges(imatge_reference)
        sum_edges_registrada = np.sum(edges_registrada)
        sum_edges_reference = np.sum(edges_reference)
        dif_edge= np.abs(edges_registrada-edges_reference)


        malla_x, malla_y = self.parametres_a_malla(parametres)

        mx_col_post, my_col_post = malla_x[:, 1:], malla_y[:, 1:]
        mx_fila_post, my_fila_post = malla_x[1:, :], malla_y[1:, :]

        d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
        d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1, :]), 2) + np.power((my_fila_post - malla_y[0:-1, :]), 2))
        sd1 = np.std(d1)
        sd2 = np.std(d2)

        residuals_info_mutua = 1 - info_mutua(imatge_reference,imatge_registrada,5)

        residuals_regularizacio = np.asarray([sd1, sd2])
        residuals_edge = np.sum(dif_edge)


        beta = 1-gamma-chi
        return np.concatenate([beta*residuals_info_mutua/np.sum(residuals_info_mutua),
         gamma*residuals_regularizacio,
         [chi *residuals_edge/(sum_edges_reference+ (sum_edges_registrada - sum_edges_reference))] ])

    def montecarlo(self, malla_vector, imatge_input, imatge_reference, path_carpeta_experiment,fitxer_sortida,
                   nombre_execucions, perturbacio, gamma, chi):
        nx = self.nx
        ny = self.ny


        #enlloc de quedarme tan sols amb el valor mínim i els seus corresponents paràmetre vull que em guardi
        #la informació respectiva als tres valors més petits. Per si així arrib a millors resultats.
        valors_optims = [0, 0, 0]
        parametres_optims = [0, 0, 0]
        execucio = -1

        for num_exec in range(1, nombre_execucions):

            execucio = execucio + 1
            factor_perturbacio = 0 if num_exec == 1 else perturbacio
            malla_x, malla_y, Coord_originals_x, Coord_originals_y = self.perturbar_malla_aleatoriament(
                malla_vector, imatge_input,
                factor_perturbacio)

            # Visualitza corregistre inicial (malla amb perturbació aleatòria)
            '''
            imatge_registrada_input = transformar(imatge_input,corregistre,np.concatenate([malla_x.flatten(), malla_y.flatten()]))

            visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla inicial aleatòria',
                             path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_original{nx}')

            '''
            funcio_min_residus = lambda x: self.residus(x, imatge_input, imatge_reference, gamma, chi)
            resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                     diff_step = None, gtol=1e-12, xtol=1e-13, ftol=1e-13,
                                     method='lm', verbose=2)

            # funcio_min_escalar = lambda x: np.sum(residus(x, imatge_reference, imatge_input, corregistre, nx, ny, sigma)**2)
            # resultat_opcio1 = minimize(funcio_min_escalar, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
            #                            method='L-BFGS-B',
            #                            # options={'eps': 0.002, 'gtol': 1e-14}
            #                            )

            # Visualitzar resultat
            parametres = resultat.x
            val_parametres = resultat.cost
            residus = funcio_min_residus(parametres)
            m = self.dim_imatge[0] * self.dim_imatge[1]

            sd_malla = residus[-3:-1]
            contorn = residus[-1]


            fitxer_sortida.write(f'''
            \n{gamma,chi}:{num_exec}:
            regularitzacio malla : {sd_malla},
            contorn : {contorn}\n
            ''')
            imatge_registrada_input = self.transformar(imatge_input, parametres)


            '''
            malla_x,malla_y = parametres_a_malla(parametres,nx+1,ny+1)
            visualitza_malla(imatge_registrada_input, malla_x, malla_y,'malla optima',
                             path_guardar=f'{path_carpeta_experiment}/{num_exec:02d}_malla_optima{nx}')
            '''


            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_imatge_registrada{nx}_{val_parametres}.png',
                   imatge_registrada_input)

            #edges

            edges_registrada = self.edges(imatge_registrada_input)
            edges_reference = self.edges(imatge_reference)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_contorn_registrada_{nx}.png', edges_registrada)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_contorn_reference_{nx}.png', edges_reference)

            #valors i parametres optims
            if execucio < 3:
                valors_optims[execucio] = val_parametres
                parametres_optims[execucio] = parametres

            if execucio == 2:  # ordenar els valors i parametres associats.
                m1 = list(valors_optims)
                m2 = list(parametres_optims)

                valors_optims = sorted(valors_optims)
                for j in range(0, 3):
                    parametres_optims[j] = m2[m1.index(valors_optims[j])]

            if execucio > 2:
                if val_parametres < valors_optims[0]:
                    valors_optims[1:] = valors_optims[0:-1]
                    valors_optims[0] = val_parametres

                    parametres_optims[1:] = parametres_optims[0:-1]
                    parametres_optims[0] = parametres
            # if rmse_gaussian < min:
            #    min = rmse_gaussian
            #    parametres_optims = parametres

        return valors_optims, parametres_optims

    def guardar_millor_imatge_registrada(self, imatge_input, imatge_reference, malla_vector,
                                         path_carpeta_experiment,fitxer_sortida,
                                         iter, perturbacio, gamma, chi):
        millors3resultats = self.montecarlo(malla_vector, imatge_input, imatge_reference,
                                            path_carpeta_experiment,fitxer_sortida,
                                            iter, perturbacio, gamma, chi)
        valors_optims = millors3resultats[0]
        parametres_optims = millors3resultats[1][0]

        mx = self.nx + 1
        my = self.ny + 1
        millor_malla_preliminar = self.parametres_a_malla(parametres_optims)

        imatge_registrada = self.transformar(imatge_input, parametres_optims)
        #visualitza_malla(imatge_registrada, millor_malla_preliminar[0], millor_malla_preliminar[1],
        #                 f'malla imatge registrada optima {mx},{my}',
        #                 f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
        #rmse = RMSE(imatge_reference, imatge_registrada)
        imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{valors_optims[0]}.png',
               imatge_registrada)
        return millor_malla_preliminar

#imatge_ref, imatge_input = pass
#model_transformar = Rescala()
#model_transformar.find_best_transform(imatge_ref, imatge_input)
#model_transformar.visualize_transform()

#imatge_input_transformada = model_transformar.apply_transform(imatge_input)

class trans_prova(BaseTransform):
    def  apply_transform(self, reference_image, input_image):
        from scipy import ndimage
        return ndimage.rotate(input_image, 30)
    '''
    és simplement una rotació per fer proves utilitzant una transformada. no se si es exactament com ho hauria de definir però ho pos així.
    '''
