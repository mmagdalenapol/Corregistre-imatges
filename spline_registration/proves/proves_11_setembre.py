import numpy as np
from skimage.io import imread
import random
from spline_registration.transform_models import ElasticTransform_SSD
from spline_registration.utils import  create_results_dir,rescalar_imatge, color_a_grisos

mida_malla =np.asarray([[2,2],[4,4],[8,8]])
pixels_per_vertex = np.asarray([20, 20])


imatge_input_orig = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_input.jpg')
imatge_ref_orig = imread('/Users/mariamagdalenapolpujadas/Desktop/universitat/tfg/GITHUB/spline_registration/proves/dog_reference.jpg')
dim_original = imatge_input_orig.shape
imatge_input = rescalar_imatge(imatge_input_orig, mida_malla[0], pixels_per_vertex)
imatge_reference = rescalar_imatge(imatge_ref_orig, mida_malla[0], pixels_per_vertex)
dim_imatge = imatge_input.shape

input = color_a_grisos(imatge_input)
reference = color_a_grisos(imatge_reference)
path_carpeta_experiment = create_results_dir(f'experiments ')
fitxer_sortida = open(f'{path_carpeta_experiment}/descripcio prova.txt', "w")
iteracions = [100,5,5]


random.seed(67)
corregistre = ElasticTransform_SSD(mida_malla, dim_imatge)
parametres_optims= corregistre.find_best_transform( input,reference,path_carpeta_experiment,fitxer_sortida,iteracions)

# passam a la escala de la imatge original el millor resultat
parametres_redimensionats = np.concatenate([dim_original[0] / dim_imatge[0] * parametres_optims[0],
                                            dim_original[1] / dim_imatge[1] * parametres_optims[1]])

corregistre4 = ElasticTransform_SSD(mida_malla, dim_original)
imatge_registrada = corregistre4.apply_transform(imatge_input_orig,parametres_redimensionats)
#visualitzar el millor corregistre

corregistre4.visualize_transform(imatge_registrada,imatge_ref_orig,parametres_redimensionats,path_carpeta_experiment)