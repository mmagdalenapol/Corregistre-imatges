import logging
import numpy as np
import random

from spline_registration.databases import anhir,cerca_imatge_anhir
from spline_registration.losses import SSD, info_mutua
from spline_registration.transform_models import ElasticTransform_SSD, ElasticTransform_IM
from spline_registration.utils import create_results_dir, rescalar_imatge, color_a_grisos, visualize_side_by_side
from skimage.io import imread, imsave

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    random.seed(106)
    log.info('Create directory to store results')
    results_dir = create_results_dir('mammary-gland2 im 106')

    inp_img = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_61-HE_A4926-4L.jpg'))
    ref_img = imread(cerca_imatge_anhir (anhir(),'mammary-gland_2', 's2_67-HE_A4926-4L'))

    inp_img = inp_img[1343:1918,255:830]
    ref_img = ref_img[1125:1700,255:830]
    imsave(f'{results_dir}/imatge_input.png', inp_img)
    imsave(f'{results_dir}/imatge_reference.png', ref_img)

    mida_malla = np.asarray([[2, 2], [4, 4], [8, 8]])
    pixels_per_vertex = np.asarray([20, 20])
    dim_original = inp_img.shape

    #inp_img = inp_img[0: int(dim_original[0] / 4), 0: int(dim_original[1] / 4)]
    #ref_img = ref_img[0: int(dim_original[0] / 4), 0: int(dim_original[1] / 4)]
    #dim_original = inp_img.shape

    # reescalam i passam a escala de grisos
    imatge_input = rescalar_imatge(inp_img, mida_malla[0], pixels_per_vertex)
    imatge_reference = rescalar_imatge(ref_img, mida_malla[0], pixels_per_vertex)
    dim_imatge = imatge_input.shape
    # input_img = color_a_grisos(imatge_input)
    # reference_img = color_a_grisos(imatge_reference)
    input_img = imatge_input
    reference_img = imatge_reference
    fitxer_sortida = open(f'{results_dir}/descripcio prova.txt', "w")
    iteracions = [100, 20, 5]

    # transform1 = ElasticTransform_SSD(mida_malla, dim_imatge)
    transform1 = ElasticTransform_IM(mida_malla, dim_imatge)
    parametres_optims = transform1.find_best_transform(input_img, reference_img, results_dir,
                                                           fitxer_sortida, iteracions)

    log.info('Visualize the results')

    parametres_redimensionats = np.concatenate([dim_original[0] / dim_imatge[0] * parametres_optims[0],
                                                    dim_original[1] / dim_imatge[1] * parametres_optims[1]])

    # transform = ElasticTransform_SSD(mida_malla, dim_original)
    transform = ElasticTransform_IM(mida_malla, dim_original)
    tfd_img = transform.apply_transform(inp_img, parametres_redimensionats)
    error_ssd = SSD(ref_img, tfd_img)
    error_IM = np.sum(info_mutua(tfd_img, ref_img, 5))
    transform.visualize_transform(tfd_img, ref_img, parametres_redimensionats, results_dir, error_IM)
    visualize_side_by_side(ref_img, tfd_img, title=f'SSD: {error_ssd}, IM: {error_IM}')
    log.info('Store the results')
