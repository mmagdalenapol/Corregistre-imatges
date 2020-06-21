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

    def colors_transform(self, reference_image, A):
        '''
         A conté les coordenades d'on se suposa que va cada pixel de la imatge corregistrada.
        Aquestes coordenades poden ser decimals i per això el valor del color depèn del color dels 4 pixels més propers.

        #si li introduim A com un array_like with shape (n,) on n= nx*ny*2.
        #ara A es és una matriu de dimensió (2, nx*ny)
        # A[0] conté les coordenades x, A[1] conté les coordenades y.
        '''

        A = np.array([A[0:(nx * ny)], A[nx * ny:(2 * nx * ny)]])
        '''
        Abans de res feim uns quantes adaptacions inicials a A. 
        -No hi ha coordenades negatives per tant qualsevol element negatiu el passam a nes 0.
        -La coordenada x no pot ser major que les files de la imatge de referència.
        -La coordenada y no pot ser major que les columnnes de la imatge de referència.
        '''

        A = np.maximum(A, 0)
        A[0] = np.minimum(A[0], reference_image.shape[0] - 1)
        A[1] = np.minimum(A[1], reference_image.shape[1] - 1)

        X = A[0]
        Y = A[1]
        Ux = np.floor(X).astype(int)
        Vy = np.floor(Y).astype(int)

        a = X - Ux
        b = Y - Vy

        # si a és 0 no hauriem de tenir en compte Ux+1 tan sols Ux i si b es 0 no he m de tenir en compte Vy+1 tan sols Vy
        M = np.array([reference_image[Ux, Vy][:, 0], reference_image[Ux, Vy][:, 1], reference_image[Ux, Vy][:, 2]])
        B = np.array(
            [reference_image[Ux + 1, Vy][:, 0], reference_image[Ux + 1, Vy][:, 1], reference_image[Ux + 1, Vy][:, 2]])
        C = np.array(
            [reference_image[Ux, Vy + 1][:, 0], reference_image[Ux, Vy + 1][:, 1], reference_image[Ux, Vy + 1][:, 2]])
        D = np.array([reference_image[Ux + 1, Vy + 1][:, 0], reference_image[Ux + 1, Vy + 1][:, 1],
                      reference_image[Ux + 1, Vy + 1][:, 2]])

        color = (a - 1) * (b - 1) * M + a * (1 - b) * B + (1 - a) * b * C + a * b * D

        return color

    def malla(self, input_image):
        #inicialitzam la malla
        nx, ny = (20, 20)
        malla = np.mgrid[0:input_image.shape[0]:round(input_image.shape[0] / nx),
                0:input_image.shape[1]:round(input_image.shape[1] / ny)]
        malla_x = malla[0]  # inicialitzam a on van les coordenades x a la imatge_reference
        malla_y = malla[1]  # inicialitzam a on van les coordenades y a la imatge_reference

        Mx = malla_x.ravel()
        My = malla_y.ravel()
        M = np.concatenate((Mx, My), axis=0)

        imatge_malla_input = self.colors_transform(input_image, M)
        imatge_malla_input = np.hsplit(imatge_malla_input, nx * ny)
        imatge_malla_input = np.array(imatge_malla_input)
        imatge_malla_input = (imatge_malla_input.ravel()).reshape(nx, ny, 3)

        return M,imatge_malla_input


    def apply_transform(self,reference_image,input_image ):

        return





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
