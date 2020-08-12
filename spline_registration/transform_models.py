from skimage.transform import resize
from spline_registration.utils import imatge_vec
import numpy as np


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
    def __init__(self):
        self.dim_imatge = None
        self.A = None
        self.nx=2
        self.ny=2

    def malla_inicial(self, imatge_input):
        nx = self.nx
        ny = self.ny

        delta = [int(imatge_input.shape[0]/nx) + 1, int(imatge_input.shape[1]/ny) + 1]

        '''
        el +1 ens permet assegurar que la darrera fila/columna de la malla estan defora de la imatge.
        Ja que així creant aquests punts ficticis a fora podem interpolar totes les posicions de la imatge. 
        Ara la malla serà (nx+1)*(ny+1) però la darrera fila i la darrera columna com he dit són per tècniques.
        '''
        malla = np.mgrid[ 0: (nx+1)*delta[0] :delta[0], 0: (ny+1)*delta[1]:delta[1]]

        malla_x = malla[0]  # inicialitzam a on van les coordenades x a la imatge_reference
        malla_y = malla[1]  # inicialitzam a on van les coordenades y a la imatge_reference

        coordenadesx = np.arange(0, (nx + 1) * delta[0], delta[0])
        coordenadesy = np.arange(0, (ny + 1) * delta[1], delta[1])

        malla_vector = np.concatenate((malla_x.ravel(), malla_y.ravel()), axis=0)

        return malla_vector

    def posicio(self, x, y, malla_x, malla_y,nx,ny):
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

        delta = [int((x[-1]+1) / nx) + 1, int((y[-1] + 1) / ny) + 1]

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

        x = np.arange(imatge.shape[0])
        y = np.arange(imatge.shape[1])

        Coord_originals_x, Coord_originals_y = np.meshgrid(x, y)
        Coord_originals_x = Coord_originals_x.ravel()
        Coord_originals_y = Coord_originals_y.ravel()

        registered_image = np.zeros_like(imatge)
        registered_image[Coord_originals_x, Coord_originals_y] = imatge[coord_desti[0], coord_desti[1]]
        return registered_image

    def colors_transform_nearest_neighbours(self,imatge_reference, Coordenades_desti):
            Coordenades_desti = np.round( Coordenades_desti).astype('int')  # Discretitzar
            Coordenades_desti = np.maximum(Coordenades_desti, 0)
            Coordenades_desti[0] = np.minimum(Coordenades_desti[0], imatge_reference.shape[0] - 1)
            Coordenades_desti[1] = np.minimum(Coordenades_desti[1], imatge_reference.shape[1] - 1)

            registered_image = imatge_reference[Coordenades_desti[0], Coordenades_desti[1]]
            registered_image = registered_image.reshape(imatge_reference.shape, order='F')

            return registered_image

    def find_best_transform(self, reference_image, input_image):

        return None

    def apply_transform(self,reference_image,input_image ):

        return None





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
