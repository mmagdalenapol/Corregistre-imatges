import logging
import numpy as np
import random

from spline_registration.databases import test_ca, test_gavina
from spline_registration.losses import SSD, info_mutua
from spline_registration.transform_models import ElasticTransform_SSD, ElasticTransform_IM
from spline_registration.utils import create_results_dir, rescalar_imatge, color_a_grisos

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    random.seed(106)
    log.info('Create directory to store results')
    results_dir = create_results_dir('Prova 106')

    for sample in test_gavina():
        name = sample['name']
        inp_img = sample['input']
        ref_img = sample['reference']
        log.info(f'Processing image {name}')

        mida_malla = np.asarray([[2, 2], [4, 4], [8, 8]])
        pixels_per_vertex = np.asarray([20, 20])
        dim_original = inp_img.shape

        # reescalam i passam a escala de grisos
        imatge_input = rescalar_imatge(inp_img, mida_malla[0], pixels_per_vertex)
        imatge_reference = rescalar_imatge(ref_img, mida_malla[0], pixels_per_vertex)
        dim_imatge = imatge_input.shape
        # input_img = color_a_grisos(imatge_input)
        # reference_img = color_a_grisos(imatge_reference)
        input_img = imatge_input
        reference_img = imatge_reference
        fitxer_sortida = open(f'{results_dir}/descripcio prova.txt', "w")
        iteracions = [100, 5, 5]

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
        # error = SSD(ref_img, tfd_img)
        error = np.sum(info_mutua(ref_img, tfd_img, 5))
        transform.visualize_transform(tfd_img, ref_img, parametres_redimensionats, results_dir, error)

        log.info('Store the results')
