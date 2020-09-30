import logging

from spline_registration.databases import anhir_test
from spline_registration.losses import always_return_zero
from spline_registration.transform_models import DoNothingTransform
from spline_registration.utils import create_results_dir, visualize_side_by_side

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


if __name__ == '__main__':

    log.info('Create directory to store results')
    results_dir = create_results_dir('Prova')

    for sample in anhir_test():
        name = sample['name']
        inp_img = sample['input']
        ref_img = sample['reference']

        log.info(f'Processing image {name}')
        transform = DoNothingTransform(loss_function=always_return_zero)
        transform.find_best_transform(reference_image=ref_img, input_image=inp_img)

        log.info('Visualize the results')
        tfd_img = transform.apply_transform(inp_img)
        error = always_return_zero(reference_image=ref_img, transformed_image=tfd_img)
        visualize_side_by_side(ref_img, tfd_img, title=f'Error: ')

        log.info('Store the results')
        # ...
