from skimage.filters import gaussian
from skimage.io import imsave
from spline_registration.utils import coordenades_originals, visualitza_malla, color_a_grisos, ampliacio_malla
import numpy as np
import random
from scipy.optimize import least_squares
from spline_registration.losses import SSD, info_mutua
from skimage import feature
from PIL import Image


class BaseTransform:
    def __init__(self, mida_malla, dim_imatge):
        self.dim_imatge = dim_imatge
        self.nx = mida_malla[:, 0]
        self.ny = mida_malla[:, 1]
        self.perturbacio = None
        self.diff_step = None
        self.chi = None
        self.gamma = None

    def imatge_gaussian(self, imatge, multichanel=True):
        sigma = (imatge.shape[0] / 3 + imatge.shape[1] / 3) * 1 / 5
        imatge_gaussian = gaussian(imatge, sigma=sigma, multichannel=multichanel)
        return imatge_gaussian

    def parametres_a_malla(self, parametres, i):
        files = self.nx[i] + 1
        columnes = self.ny[i] + 1
        malla_x = parametres[0: files * columnes].reshape(files, columnes)
        malla_y = parametres[files*columnes: 2*files*columnes].reshape(files, columnes)
        return malla_x, malla_y

    def malla_inicial(self, i):
        nx = self.nx[i]
        ny = self.ny[i]
        delta = [int(self.dim_imatge[0] / nx) + 1, int(self.dim_imatge[1] / ny) + 1]

        '''
        el +1 ens permet assegurar que la darrera fila/columna de la malla estan defora de la imatge.
        Ja que així creant aquests punts ficticis a fora podem interpolar totes les posicions de la imatge. 
        Ara la malla serà (nx+1)*(ny+1).
        '''
        malla = np.mgrid[0: (nx + 1) * delta[0]: delta[0], 0: (ny + 1) * delta[1]: delta[1]]
        malla_x = malla[0]  # inicialitzam a on van les coordenades x a la imatge_reference
        malla_y = malla[1]  # inicialitzam a on van les coordenades y a la imatge_reference
        malla_vector = np.concatenate((malla_x.ravel(), malla_y.ravel()), axis=0)

        return malla_vector

    def perturbar_malla_aleatoriament(self, malla_vector, imatge_input, iteracions, i):

        malla_x, malla_y = self.parametres_a_malla(malla_vector, i)
        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge_input)

        nx = self.nx[i]
        ny = self.ny[i]
        delta = [int(self.dim_imatge[0] / nx) + 1, int(self.dim_imatge[1] / ny) + 1]
        perturbacio = self.perturbacio

        epsilon1 = np.zeros(malla_x.shape)
        epsilon2 = np.zeros(malla_y.shape)
        if iteracions > 0:
            for i in range(0, malla_x.shape[0]):
                for j in range(0, malla_x.shape[1]):
                    epsilon1[i, j] = random.randrange(-int(delta[0] * perturbacio), int(delta[0] * perturbacio))
            for i in range(0, malla_y.shape[0]):
                for j in range(0, malla_y.shape[1]):
                    epsilon2[i, j] = random.randrange(-int(delta[1] * perturbacio), int(delta[1] * perturbacio))

        malla_x = malla_x + epsilon1
        malla_y = malla_y + epsilon2
        return malla_x, malla_y, Coord_originals_x, Coord_originals_y

    def posicio(self, x, y, malla_x, malla_y, i):
        # s val 0 quan la x està a coordenadesx
        # t val 0 quan la y està a coordenadesy
        # i index de la posició més pròxima per davall de la coordenada x a la malla
        # j index de la posició més pròxima per davall de la coordenada y a la malla

        nx = self.nx[i]
        ny = self.ny[i]
        delta = [int(self.dim_imatge[0] / nx) + 1, int(self.dim_imatge[1] / ny) + 1]

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

    def imatge_transformada(self, imatge_input, coord_desti):
        '''
        Introduim la imatge_input i les coordenades a les quals es mouen les originals després d'aplicar l'interpolació.
        El que volem es tornar la imatge registrada que tengui a les coordenades indicades els colors originals:

        Per fer-ho definesc una imatge registrada (inicialment tota negre) i a les coordenades del destí
        anar enviant els colors originals.
        '''

        coord_desti = np.round(coord_desti).astype('int')  # Discretitzar
        coord_desti = np.maximum(coord_desti, 0)
        coord_desti[0] = np.minimum(coord_desti[0], imatge_input.shape[0] - 1)
        coord_desti[1] = np.minimum(coord_desti[1], imatge_input.shape[1] - 1)

        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge_input)

        registered_image = np.zeros_like(imatge_input)
        registered_image[Coord_originals_x, Coord_originals_y] = imatge_input[coord_desti[0], coord_desti[1]]
        return registered_image

    def transformar(self, imatge, parametres, i):

        malla_x, malla_y = self.parametres_a_malla(parametres, i)
        Coord_originals_x, Coord_originals_y = coordenades_originals(imatge)

        Coordenades_desti = self.posicio(Coord_originals_x, Coord_originals_y, malla_x, malla_y, i)

        return self.imatge_transformada(imatge, Coordenades_desti)

    def montecarlo(self, malla_vector, imatge_input, imatge_reference, path_carpeta_experiment, fitxer_sortida,
                   nombre_execucions, i):
        nx = self.nx[i]
        ny = self.ny[i]
        diff_step = self.diff_step
        gamma = self.gamma
        chi = self.chi

        valors_optims = 20
        parametres_optims = 0

        for num_exec in range(1, nombre_execucions):

            malla_x, malla_y, Coord_originals_x, Coord_originals_y = self.perturbar_malla_aleatoriament(
                                                                        malla_vector, imatge_input, num_exec, i)

            funcio_min_residus = lambda x: self.residus(x, imatge_input, imatge_reference, i)

            resultat = least_squares(funcio_min_residus, x0=np.concatenate([malla_x.flatten(), malla_y.flatten()]),
                                     diff_step=diff_step, gtol=1e-12, xtol=1e-13, ftol=1e-13,
                                     method='lm', verbose=2)

            parametres = resultat.x
            val_parametres = resultat.cost
            residus = funcio_min_residus(parametres)

            residuals_error = residus[0:-3]/(1-gamma-chi)
            sum_residuals = np.sum(residuals_error**2)
            min = np.min(residuals_error)
            max = np.max(residuals_error)
            mean = np.mean(residuals_error)
            sd_malla = residus[-3:-1]/gamma
            contorn = residus[-1]/chi

            fitxer_sortida.write(f'''{num_exec}:
            \n {resultat}\n
            \n{gamma,chi}:
            min,max,mean dels residus colors originals,suma:{min,max,mean,sum_residuals},
            regularitzacio malla : {sd_malla},
            contorn : {contorn}\n
            ''')
            imatge_registrada_input = self.transformar(imatge_input, parametres, i)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_imatge_registrada{nx,ny}_{val_parametres}.png',
                   imatge_registrada_input)

            # edges
            edges_registrada = self.edges(imatge_registrada_input)
            imsave(f'{path_carpeta_experiment}/{num_exec:02d}_contorn_registrada_{nx,ny}.png', edges_registrada)

            if val_parametres < valors_optims:
                valors_optims = val_parametres
                parametres_optims = parametres

        return valors_optims, parametres_optims

    def guardar_millor_imatge_registrada(self, imatge_input, imatge_reference, malla_vector,
                                         path_carpeta_experiment, fitxer_sortida, iteracions, i):

        millorsresultats = self.montecarlo(malla_vector, imatge_input, imatge_reference,
                                           path_carpeta_experiment, fitxer_sortida, iteracions, i)
        valors_optims = millorsresultats[0]
        parametres_optims = millorsresultats[1]

        mx = self.nx[i] + 1
        my = self.ny[i] + 1
        millor_malla_preliminar = self.parametres_a_malla(parametres_optims, i)

        imatge_registrada = self.transformar(imatge_input, parametres_optims, i)
        imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{valors_optims}.png',
               imatge_registrada)
        return millor_malla_preliminar

    def find_best_transform(self, input_image, reference_image, path_carpeta_experiment, fitxer_sortida, iteracions):
        malla_vector = self.malla_inicial(0)

        for i in range(0, 2):
            millor_malla_preliminar = self.guardar_millor_imatge_registrada(input_image, reference_image, malla_vector,
                                                                            path_carpeta_experiment, fitxer_sortida,
                                                                            iteracions[i], i)

            malla_ampliada = ampliacio_malla(millor_malla_preliminar[0], millor_malla_preliminar[1])
            malla_vector = [malla_ampliada[0].ravel(), malla_ampliada[1].ravel()]
            malla_vector = np.asarray(malla_vector).ravel()

        millor_malla_preliminar = self.guardar_millor_imatge_registrada(input_image, reference_image, malla_vector,
                                                                        path_carpeta_experiment, fitxer_sortida,
                                                                        iteracions[2], 2)
        parametres_optims = np.asarray([millor_malla_preliminar[0].ravel(), millor_malla_preliminar[1].ravel()])

        return parametres_optims

    def apply_transform(self, input_image, parametres):
        imatge_registrada = self.transformar(input_image, parametres, 2)
        return imatge_registrada

    def visualize_transform(self, registered_image, reference_image, parametres, path_carpeta_experiment, error):

        imsave(f'{path_carpeta_experiment}/imatge_reference.png', reference_image)

        mx, my = self.nx[-1]+1, self.ny[-1]+1
        millor_malla = self.parametres_a_malla(parametres, 2)

        visualitza_malla(registered_image, millor_malla[0], millor_malla[1],
                         f'malla imatge registrada optima {mx, my}',
                         f'{path_carpeta_experiment}/malla_imatge_registrada{mx, my}.png')
        imsave(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{error}.png',
               registered_image)
        im1 = Image.open(f'{path_carpeta_experiment}/imatge_reference.png').convert('L')
        im2 = Image.open(f'{path_carpeta_experiment}/imatge_registrada_{mx, my}_{error}.png').convert('L')
        im = Image.blend(im1, im2, 0.5)
        path_imatge_blend = f'{path_carpeta_experiment}/imatge_blend.png'
        im.save(path_imatge_blend)


class ElasticTransform_SSD(BaseTransform):
    def __init__(self, mida_malla, dim_imatge):
        self.dim_imatge = dim_imatge
        self.A = None
        self.nx = mida_malla[:, 0]
        self.ny = mida_malla[:, 1]
        self.diff_step = None
        self.gamma = 0.2
        self.chi = 0.08
        self.perturbacio = 1/5

    def edges(self, imatge):
        sigma = (imatge.shape[0] / 10 + imatge.shape[1] / 10) * 1 / 5
        imatge_gaussian = gaussian(imatge, sigma=sigma, multichannel=False)
        edges = feature.canny(imatge_gaussian)
        edges = np.where(edges == True, 1, 0)
        return edges

    def residus(self, parametres, imatge_input, imatge_reference, i):
        gamma = self.gamma
        chi = self.chi
        beta = 1 - gamma - chi

        # definim les imatges registrada inicial, la registrada amb filtratge gaussià
        # i la referència amb filtratge gaussià
        regist_img = self.transformar(imatge_input, parametres, i)
        # enviam les coord_originals de la imatge input a les coor_desti
        gaus_reg_img = self.imatge_gaussian(regist_img)
        gaus_ref_img = self.imatge_gaussian(imatge_reference)

        # calculam el factor de regularització que depen dels edges
        regist_edges = self.edges(regist_img)
        ref_edges = self.edges(imatge_reference)
        sum_regist_edges = np.sum(regist_edges)
        sum_ref_edges = np.sum(ref_edges)
        dif_edge = np.abs(regist_edges - ref_edges)
        residuals_edge = np.sum(dif_edge)

        # calculam el factor de regularització que depen dels punts de la malla
        malla_x, malla_y = self.parametres_a_malla(parametres, i)

        mx_col_post, my_col_post = malla_x[:, 1:], malla_y[:, 1:]
        mx_fila_post, my_fila_post = malla_x[1:, :], malla_y[1:, :]

        d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
        d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1, :]), 2) + np.power((my_fila_post - malla_y[0:-1, :]), 2))
        sd1 = np.std(d1)
        sd2 = np.std(d2)
        residuals_regularizacio = np.asarray([sd1, sd2])

        # calculam els residus obtinguts de comparar pixel a pixel les imatges tant originals com gaussianes
        residuals_ssd = np.power((regist_img - imatge_reference).flatten(), 2)
        residuals_gaus_ssd = np.power((gaus_reg_img - gaus_ref_img).flatten(), 2)

        # return np.concatenate([residuals_gaus_ssd / sum(residuals_gaus_ssd), gamma * residuals_regularizacio])
        den = sum(residuals_ssd + residuals_gaus_ssd)
        if den == 0:
            den = 1

        return np.concatenate([beta * residuals_ssd / den,
                               beta * residuals_gaus_ssd / den,
                               gamma * residuals_regularizacio,
                               [chi * residuals_edge / sum_regist_edges]])


class ElasticTransform_IM (BaseTransform):
    def __init__(self, mida_malla, dim_imatge):
        self.dim_imatge = dim_imatge
        self.A = None
        self.nx = mida_malla[:, 0]
        self.ny = mida_malla[:, 1]
        self.diff_step = None
        self.gamma = 0.1
        self.chi = 0.08
        self.perturbacio = 1 / 5

    def edges(self, imatge):
        edges = feature.canny(color_a_grisos(imatge))
        edges = np.where(edges == True, 1, 0)
        return edges

    def residus(self, parametres, imatge_input, imatge_reference, i):
        gamma = self.gamma
        chi = self.chi

        imatge_registrada = self.transformar(imatge_input, parametres, i)  # enviam les coord_originals de la imatge input a les coor_desti

        edges_registrada = self.edges(imatge_registrada)
        edges_reference = self.edges(imatge_reference)
        sum_edges_registrada = np.sum(edges_registrada)
        sum_edges_reference = np.sum(edges_reference)
        dif_edge = np.abs(edges_registrada-edges_reference)

        malla_x, malla_y = self.parametres_a_malla(parametres, i)

        mx_col_post, my_col_post = malla_x[:, 1:], malla_y[:, 1:]
        mx_fila_post, my_fila_post = malla_x[1:, :], malla_y[1:, :]

        d1 = np.sqrt(np.power((mx_col_post - malla_x[:, 0:-1]), 2) + np.power((my_col_post - malla_y[:, 0:-1]), 2))
        d2 = np.sqrt(np.power((mx_fila_post - malla_x[0:-1, :]), 2) + np.power((my_fila_post - malla_y[0:-1, :]), 2))
        sd1 = np.std(d1)
        sd2 = np.std(d2)

        residuals_info_mutua_orig = (np.log(np.exp(1))/np.exp(1) - info_mutua(imatge_reference, imatge_registrada, 5))
        residuals_regularizacio = np.asarray([sd1, sd2])
        residuals_edge = np.sum(dif_edge)/(sum_edges_reference + sum_edges_registrada)

        beta = 1 - gamma - chi
        return np.concatenate([beta * 100 * residuals_info_mutua_orig / ((5 ** 3) ** 2),
                               gamma * residuals_regularizacio,
                               [chi * residuals_edge]])




