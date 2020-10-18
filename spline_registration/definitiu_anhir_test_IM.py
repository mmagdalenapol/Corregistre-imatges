import logging
import numpy as np
import random

from spline_registration.databases import anhir_test
from spline_registration.losses import RMSE,info_mutua
from spline_registration.transform_models import ElasticTransform_IM
from spline_registration.utils import create_results_dir, rescalar_imatge
from skimage.io import imsave
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':

    i=1
    for sample in anhir_test():
        random.seed(i)
        log.info('Create directory to store results')
        results_dir = create_results_dir(f'Prova anhir_test(3) IM {i}')
        name = sample['name']
        inp_img = sample['input']
        ref_img = sample['reference']
        log.info(f'Processing image {name}')

        mida_malla = np.asarray([[2, 2], [4, 4], [8, 8]])
        pixels_per_vertex = np.asarray([20, 20])
        dim_original = inp_img.shape

        inp_img1 = inp_img[int(dim_original[0] / 4): -int(dim_original[0] / 4), int(dim_original[1] / 4): -int(dim_original[1] / 4)]
        ref_img1 = ref_img[int(dim_original[0] / 4): -int(dim_original[0] / 4), int(dim_original[1] / 4): -int(dim_original[1] / 4)]


        inp_img2 = inp_img[2*int(dim_original[0] / 4): -1, 0: 2*int(dim_original[1] / 4)]
        ref_img2 = ref_img[2*int(dim_original[0] / 4): -1, 0: 2*int(dim_original[1] / 4)]

        inp_img3 = inp_img[0: 2*int(dim_original[0] / 4), 0: 2*int(dim_original[1] / 4)]
        ref_img3 = ref_img[0: 2*int(dim_original[0] / 4), 0: 2*int(dim_original[1] / 4)]

        inp_img = inp_img3
        ref_img = ref_img3

        dim_original = inp_img.shape
        imsave(f'{results_dir}/imatge_input.png', inp_img)
        imsave(f'{results_dir}/imatge_reference.png', ref_img)

        # reescalam
        imatge_input = rescalar_imatge(inp_img, mida_malla[0], pixels_per_vertex)
        imatge_reference = rescalar_imatge(ref_img, mida_malla[0], pixels_per_vertex)
        dim_imatge = imatge_input.shape
        input_img = imatge_input
        reference_img = imatge_reference
        fitxer_sortida = open(f'{results_dir}/descripcio prova.txt', "w")
        iteracions = [100, 20, 5]

        transform1 = ElasticTransform_IM(mida_malla, dim_imatge)
        parametres_optims = transform1.find_best_transform(input_img, reference_img, results_dir,
                                                           fitxer_sortida, iteracions)

        log.info('Visualize the results')

        parametres_redimensionats = np.concatenate([dim_original[0] / dim_imatge[0] * parametres_optims[0],
                                                    dim_original[1] / dim_imatge[1] * parametres_optims[1]])

        transform = ElasticTransform_IM(mida_malla, dim_original)
        tfd_img = transform.apply_transform(inp_img, parametres_redimensionats)
        error_rmse = RMSE(ref_img, tfd_img)
        error_IM = np.sum(info_mutua(tfd_img, ref_img, 5))
        transform.visualize_transform(inp_img, tfd_img, ref_img, parametres_redimensionats, results_dir, error_IM)
        fitxer_sortida.write(f'\n\n\nvalors entre imatge registrada i referència error rmse:{error_rmse},'
                             f' informació mútua: {error_IM} ')
        fitxer_sortida.write(f'\n\n\nvalors entre imatge input i referència error rmse:{RMSE(ref_img, inp_img)},'
                             f' informació mútua: {np.sum(info_mutua(inp_img, ref_img, 5))} ')

        log.info('Store the results')

