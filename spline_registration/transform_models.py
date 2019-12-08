from skimage.transform import resize

from spline_registration.utils import imatge_vec

class BaseTransform:
    def find_best_transform(self, reference_image, input_image):
        raise NotImplementedError

    def apply_transform(self, input_image):
        return input_image

    def visualize_transform(self):
        return None


class Rescala(BaseTransform):
    def __init__(self):
        self.dim_imatge = None

    def find_best_transform(self, reference_image, input_image):
        self.dim_imatge = reference_image.shape

    def apply_transform(self, input_image):
        return resize(input_image,self.dim_imatge)


class ElasticTransform(BaseTransform):
    def __init__(self):
        self.dim_imatge = None


    #no soluciona el problema de que vagi a jocs fora de la imatge
    def colors_transform(self, reference_image, input_image):
        A = BaseTransform.apply_transform(input_image)
        #la transformacio m'hauria de donar les posicions d'on se suposa que ve cada pixel de la imatge corregistrada.
        #aquests valors poden ser decimals i per aixo el valor del color depèn dels 4 parells d'enters més pròxims.

        columna1 = A[:, :, 0]
        columna2 = A[:, :, 1]
        A21 = np.floor(A)#tots els elements enter més petit o igual
        A12 = np.ceil(A)#tots els elements enter més gran o igual
        A11 = np.ones(A.shape)
        A11[:, :, 0] = np.floor(columna1)
        A11[:, :, 1] = np.ceil(columna2)#A11 files floor i columnes ceil
        A22 = np.ones(A.shape)
        A22[:, :, 0] = np.ceil(columna1)
        A22[:, :, 1] = np.floor(columna2)#A22 files ceil i columnes floor

        #ara ho passam a vector i els nombres a enters per poder emprarlos com indexs
        A = imatge_vec(A, 2)
        A11 = imatge_vec(A11, 2).astype(int)
        A12 = imatge_vec(A12, 2).astype(int)
        A21 = imatge_vec(A21, 2).astype(int)
        A22 = imatge_vec(A22, 2).astype(int)

        # SI TOT FOSSIN NO ENTERS FUNCIONA:
        f11 = (1 - (-A11[:, 0] + A[:, 0]) * (A11[:, 1] - A[:, 1])) / 3
        f12 = (1 - (A12[:, 0] - A[:, 0]) * (A12[:, 1] - A[:, 1])) / 3
        f21 = (1 - (-A21[:, 0] + A[:, 0]) * (-A21[:, 1] + A[:, 1])) / 3
        f22 = (1 - (A22[:, 0] - A[:, 0]) * (-A22[:, 1] + A[:, 1])) / 3

        #canvis pels enters:
        # canviam les fij de les posicions on les COLUMNES són enters:
        f11[np.where(A11[:, 1] == A[:, 1])] = (1 - (-A11[:, 0] + A[:, 0]))[np.where(A11[:, 1] == A[:, 1])]
        f12[np.where(A11[:, 1] == A[:, 1])] = (1 - (A12[:, 0] - A[:, 0]))[np.where(A11[:, 1] == A[:, 1])]
        f21[np.where(A11[:, 1] == A[:, 1])] = 0
        f22[np.where(A11[:, 1] == A[:, 1])] = 0

        # canviam les fij de les posicions on les FILES són enters:
        f11[np.where(A11[:, 0] == A[:, 0])] = (1 - (A11[:, 1] - A[:, 1]))[np.where(A11[:, 0] == A[:, 0])]
        f12[np.where(A11[:, 0] == A[:, 0])] = 0
        f21[np.where(A11[:, 0] == A[:, 0])] = (1 - (-A21[:, 1] + A[:, 1]))[np.where(A11[:, 0] == A[:, 0])]
        f22[np.where(A11[:, 0] == A[:, 0])] = 0

        # si tant files com columnes són enters:
        f11[np.where(np.sum(A11 == A, axis=1) == 2)] = 1
        f12[np.where(np.sum(A11 == A, axis=1) == 2)] = 0
        f21[np.where(np.sum(A11 == A, axis=1) == 2)] = 0
        f22[np.where(np.sum(A11 == A, axis=1) == 2)] = 0

        B = (f11 * reference_image[A11[:, 0], A11[:, 1]] + f12 * reference_image[A12[:, 0], A12[:, 1]] + f21 * reference_image[
            A21[:, 0], A21[:, 1]] + f22 * reference_image[A22[:, 0], A22[:, 1]])

        return B

    def apply_transform(self, ):

        return





#imatge_ref, imatge_input = pass
#model_transformar = Rescala()
#model_transformar.find_best_transform(imatge_ref, imatge_input)
#model_transformar.visualize_transform()

#imatge_input_transformada = model_transformar.apply_transform(imatge_input)

